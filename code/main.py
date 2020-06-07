#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd
from model import *
from utils import *
from vizualization_module import *


# In[2]:


#FLAGS:
#path = './data/'
path='D:\data\\' # path to the .npz file
skip=1 # number of sequences to skip
len_of_sequences_input = 10 # length of sequence of training
len_of_sequences_output = 10 #length of sequence for predicting
temporal_kernel_size = len_of_sequences_input # temporal kernel size for creating the spatio-temporal adj.matrix
downsample_factor = 2 # downsample fps. for=2, downsample to 25fps
joints_to_consider = 17 # 17 or 32
actions_to_consider='all' #actions to train for
#['Walking','Eating','Smoking','Discussion']# list with action names to be considered for the training.alternatively, use 'all' to include all actions
returning='test' # test or train or validation
n_stgcnn_layers=3 # number of stgcnn layers
n_txpcnn_layers=4 # number of txpcnn layers
embedding_dim=20 # dimensions for the coordinates of the embedding
input_dim=3 # dimensions of the input coordinates(default=3)
n_epochs=10 # number of epochs
batch_size=128
optimizer='adam' # adam or sgd
lr=1e-03 # learning rate
train=False # whether to train or load a model_state_dict
model_name = 'model_ckpt' 
path_of_model=path+model_name # specify where to save/load, depending on the train flag
path_of_train_stats=path+'train_stats_'+model_name # save/load the train data stats
#FLAGS FOR VISUALIZATION
viz=True # if False,no visualization
subject_to_consider=['S5'] # =S5 for test set
actions_to_consider_test=['Walking','Eating','Smoking','Discussion'] #actions to visualize and test on,all or list of actions
number_of_sequences_to_visualize=2 # # of sequences to visualize for each subject,action
# NORMALIZATION
# None - no normalization, True - Mean Std normalization, False - avg_body_len normalization
normalize = True


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)


# In[4]:


actions_to_consider=all_actions(actions_to_consider)

actions_to_consider_test=all_actions(actions_to_consider_test)


# In[5]:


A = get_adj(joints_to_consider,temporal_kernel_size).to(device)

modello = Social_STGCNN_model(n_stgcnn_layers,n_txpcnn_layers,input_dim,embedding_dim,len_of_sequences_input,
                           len_of_sequences_output,temporal_kernel_size).to(device)

if optimizer=='adam':
    optimizer=optim.Adam(modello.parameters(),lr=lr)
elif optimizer=='sgd':
    optimizer=optim.SGD(modello.parameters(),lr=lr,momentum=0.9,nesterov=True)


# In[6]:


print('total number of parameters of the network is: '+str(sum(p.numel() for p in modello.parameters() if p.requires_grad)))


# In[7]:



    def trainer():
        train_loss = []
        val_loss = []
        modello.train()
        train_norm_stats_list, Dataset = create_sequences(path,skip,len_of_sequences_input,len_of_sequences_output,downsample_factor,    joints_to_consider,actions_to_consider,returning,normalize,train_normalization=normalize)
        loader_train = DataLoader(
            Dataset,
            batch_size=batch_size,
            shuffle = True,
            num_workers=0)    

        _, Dataset_val = create_sequences(path,skip,len_of_sequences_input,len_of_sequences_output,downsample_factor,
                                          joints_to_consider,actions_to_consider,'validation',normalize,train_normalization=train_norm_stats_list)

        loader_val = DataLoader(
            Dataset_val,
            batch_size=batch_size,
            shuffle = True,
            num_workers=0)                          


        for epoch in range(n_epochs):
            running_loss=0
            modello.train()
            for cnt,batch in enumerate(loader_train): 
                    batch = [tensor.to(device) for tensor in batch]
                    sequences_train=batch[0]
                    sequences_predict_gt=batch[1]
                    optimizer.zero_grad()
                    sequences_predict=modello(sequences_train,A)
                    loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt.permute(0,2,3,1),normalize,train_norm_stats_list)*1000# # both must have format (batch,T,V,C)
                    if cnt % 100 == 0:
                        print(epoch, cnt, loss.item())                
                    loss.backward()

                    optimizer.step()
                    running_loss += loss
            train_loss.append(running_loss/cnt)
            modello .eval()
            with torch.no_grad():
                running_loss=0
                for cnt,batch in enumerate(loader_val): 
                    batch = [tensor.to(device) for tensor in batch]
                    sequences_train=batch[0]
                    sequences_predict_gt=batch[1]
                    sequences_predict=modello(sequences_train,A)
                    loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt.permute(0,2,3,1),normalize,train_norm_stats_list)*1000
                    running_loss+=loss
                val_loss.append(running_loss/cnt)



        torch.save(modello.state_dict(),path_of_model)
        with open(path_of_train_stats+'.pickle', 'wb') as f:
            pickle.dump(train_norm_stats_list, f)

        modello.eval()
        
        plt.figure(2)
        plt.plot(train_loss, 'r', label='Train loss')
        plt.plot(val_loss, 'g', label='Val loss')
        plt.legend()
        plt.show()
        
        return train_norm_stats_list

    def test():
        print('Test mode')
        modello.load_state_dict(torch.load(path_of_model))
        with open(path_of_train_stats+'.pickle', 'rb') as f:
            train_stats_ = pickle.load(f)
        modello.eval()
        accum_loss=0  
        n_batches=0 # number of batches for all the sequences
        for action in actions_to_consider_test:
          #  print(action)
            running_loss=0
            train_norm_stats_list, Dataset = create_sequences(path,skip,len_of_sequences_input,len_of_sequences_output,downsample_factor,joints_to_consider,action,returning,normalize,train_normalization=train_stats_)
            train_norm_stats_list = train_stats_
            loader_test = DataLoader(
                Dataset,
                batch_size=batch_size,
                shuffle =False,
                num_workers=0)
            with torch.no_grad():
                for cnt,batch in enumerate(loader_test): 
                    batch = [tensor.to(device) for tensor in batch]
                    sequences_train=batch[0]
                    sequences_predict_gt=batch[1]
                    sequences_predict=modello(sequences_train,A)
                    loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt.permute(0,2,3,1),normalize,train_norm_stats_list)*1000
                   # print(loss)
                    running_loss+=loss
                    accum_loss+=loss

                print('loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss/cnt))
                n_batches+=cnt
        print('overall average loss in mm is: '+str(accum_loss/n_batches))


# In[8]:


if train==True:
    trainer()
else:
    test()

    
if viz==True:
    modello.eval()
    sub=seq_for_viz(path,skip,len_of_sequences_input,len_of_sequences_output,downsample_factor,joints_to_consider,subject_to_consider,
                   actions_to_consider_test,number_of_sequences_to_visualize)
    with open(path_of_train_stats+'.pickle', 'rb') as f:
        train_stats_ = pickle.load(f)
    visualize_sequences(sub,len_of_sequences_input,len_of_sequences_output,joints_to_consider,subject_to_consider
                       ,actions_to_consider_test,number_of_sequences_to_visualize,modello,A,normalize,train_stats_)


# In[ ]:




