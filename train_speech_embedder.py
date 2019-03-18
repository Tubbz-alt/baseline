#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import random
import time
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from tqdm import tqdm
from datetime import datetime
import subprocess


def train(model_path):
    FNULL = open(os.devnull, 'w')
    device = torch.device(hp.device)
    
    if hp.data.data_preprocessed:
        train_dataset = SpeakerDatasetTIMITPreprocessed(is_training=True)
        test_dataset = SpeakerDatasetTIMITPreprocessed(is_training=False)
    else:
        train_dataset = SpeakerDatasetTIMIT(is_training=True)
        test_dataset = SpeakerDatasetTIMIT(is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    
    embedder_net = SpeechEmbedder().to(device)
    if hp.train.restore:
        subprocess.call(['gsutil', 'cp', 'gs://edinquake/asr/baseline_TIMIT/model_best.pkl', model_path], stdout=FNULL, stderr=subprocess.STDOUT)
        embedder_net.load_state_dict(torch.load(model_path))
    ge2e_loss = GE2ELoss(device)
    #Both net and loss have trainable parameters
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
       
    iteration = 0
    best_validate = float('inf')
    print('***Started training at {}***'.format(datetime.now()))
    for e in range(hp.train.epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(e), leave=False, disable=False)
        embedder_net.train()
        # Iterate over the training set
        for batch_id, mel_db_batch in enumerate(progress_bar): 
            mel_db_batch = mel_db_batch.to(device)
            
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()
            
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            
            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()
            
            total_loss = total_loss + loss.item()
            iteration += 1

            # Update statistics for progress bar
            progress_bar.set_postfix(iteration=iteration, loss=loss.item(), total_loss=total_loss/(batch_id + 1))

        print('| Epoch {:03d}: total_loss {}'.format(e, total_loss))
        # Perform validation
        embedder_net.eval()
        validation_loss = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            mel_db_batch = mel_db_batch.to(device)
            
            mel_db_batch = torch.reshape(mel_db_batch, (hp.test.N*hp.test.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.test.N*hp.test.M), hp.test.N*hp.test.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
                        
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.test.N, hp.test.M, embeddings.size(1)))
            #get loss
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            validation_loss += loss.item()

        validation_loss /= len(test_loader)
        print('validation_loss: {}'.format(validation_loss))
        if validation_loss <= best_validate:
            best_validate = validation_loss
            # Save best
            filename = 'model_best.pkl'
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            subprocess.call(['gsutil', 'cp', ckpt_model_path, 'gs://edinquake/asr/baseline_TIMIT/model_best.pkl'], stdout=FNULL, stderr=subprocess.STDOUT)
        
        filename = 'model_last.pkl'
        ckpt_model_path = os.path.join(hp.train.checkpoint_dir, filename)
        torch.save(embedder_net.state_dict(), ckpt_model_path)


def test(model_path):
    
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        test_dataset = SpeakerDatasetTIMIT()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    
    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    
    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            
            # calculating EER
            diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
            
            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix>thres
                
                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)
    
                FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                /(float(hp.test.M/2))/hp.test.N)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
        
if __name__=="__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        test(hp.model.model_path)
