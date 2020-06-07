
import numpy as np
import math
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torch.autograd import Variable


def create_sequences(path,skip,len_of_sequences_input,len_of_sequences_output,downsample_factor,joints_to_consider,
                   actions_to_consider,returning,normalize,train_normalization=None):
    len_of_all_sequences=len_of_sequences_input+len_of_sequences_output
    data=np.load(path+'data_3d_h36m.npz',allow_pickle=True)['positions_3d'].item()
    sub={}
    dataset=[]
    actions_to_consider =all_actions(actions_to_consider)
    for subject, actions in data.items():
        sub[subject]={}
        for action_name, positions in actions.items():
          #  print(subject,action_name,positions.shape)
            new_positions=positions[::downsample_factor] # subsampling to 25 fps
            sub[subject][action_name]=new_positions
         #   print('subsampled_positions')
           # print(new_positions.shape)
            n_seq=int(math.ceil(new_positions.shape[0]- len_of_all_sequences+1)) # name of sequences to be created for every chunk
            seq_data=np.zeros((n_seq,len_of_all_sequences,positions.shape[1],positions.shape[2]))
            i=0
            for idx in range(0,n_seq,skip):
                curr_seq=new_positions [idx:idx+len_of_all_sequences]
                seq_data[i,:,:,:]=curr_seq
                i+=1
            if subject=='S5'and returning=='test' and action_name.split(' ')[0] in actions_to_consider:
                dataset.append(seq_data[:,:,:,:])
            elif subject=='S11'and returning=='validation' and action_name.split(' ')[0] in actions_to_consider :
                dataset.append(seq_data[:,:,:,:])

            elif subject!='S5' and subject!='S11' and returning=='train' and action_name.split(' ')[0] in actions_to_consider:
                dataset.append(seq_data[:,:,:,:])

        
    if joints_to_consider==17:
        joints_to_keep=[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        dataset=[n[:,:,joints_to_keep,:] for n in dataset]
        
    sequences=np.vstack(dataset) # concatenate the sequences of all subjects and actions 
    
    # NORMALIZATION

    if normalize == True:
        # MEAN STD NORMALIZATION
        if returning == 'train':
            #return Dataset and normalization statistics
        
            # Normalize training data and compute data mean and std
            data_mean_train, data_std_train, Dataset = \
            Mean_Std_Tensor_Normalize(sequences,len_of_sequences_input, len_of_sequences_output,\
            data_mean_train = 0, data_std_train = 0, train=True)
            norm_stats_list = [data_mean_train, data_std_train]
        else:
            #return Dataset and apply train normalization statistics
            Dataset = Mean_Std_Tensor_Normalize(sequences,len_of_sequences_input, len_of_sequences_output,\
            data_mean_train = train_normalization[0], data_std_train = train_normalization[1], train=False)
            norm_stats_list = None

    elif normalize == False:
        # AVG BODY LEN NORMALIZATION
        if returning == 'train':
            #return Dataset and normalization statistics

            # Normalize training data and compute average body length
            avg_len, Dataset = Avg_Body_Length_Tensor_Normalize\
            (sequences,len_of_sequences_input, len_of_sequences_output, \
            avg_len_train=0,train=True)
            norm_stats_list = [avg_len]
        else:
            Dataset = Avg_Body_Length_Tensor_Normalize(sequences,\
            len_of_sequences_input, len_of_sequences_output,\
            avg_len_train=train_normalization[0],train=False)
            norm_stats_list = None

    elif normalize == None:
        # NO NORMALIZATION APPLIED
        # same for train and test
        Dataset = No_normalization(sequences,len_of_sequences_input, len_of_sequences_output)
        norm_stats_list = None

    #print('Data Loaded and Normalized')    
    return norm_stats_list, Dataset
    



def normalize_A(A): # given an adj.matrix, normalize it by multiplying left and right with the degree matrix, in the -1/2 power
        
        A=A+np.eye(A.shape[0])
        
        D=np.sum(A,axis=0)
        
        
        D=np.diag(D.A1)

        
        D_inv = D**-0.5
        D_inv[D_inv==np.infty]=0
        
        #np.matmul(np.matmul(D_inv,A),D_inv) 
        return D_inv*A*D_inv




def spatio_temporal_graph(joints_to_consider,temporal_kernel_size,spatial_adjacency_matrix): # given a normalized spatial adj.matrix,creates a spatio-temporal adj.matrix

    
    number_of_joints=joints_to_consider

    spatio_temporal_adj=np.zeros((temporal_kernel_size,number_of_joints,number_of_joints))
    for t in range(temporal_kernel_size):
        for i in range(number_of_joints):
            spatio_temporal_adj[t,i,i]=1 # create edge between same body joint,for t consecutive frames
            for j in range(number_of_joints):
                if spatial_adjacency_matrix[i,j]!=0: # if the body joints are connected
                    #spatio_temporal_adj[t,i,j]=l2_norm(sequence[t,i,:],sequence[t,j,:]) # replace 1 with their distance
                    spatio_temporal_adj[t,i,j]=spatial_adjacency_matrix[i,j]
    return spatio_temporal_adj


# In[13]:


def get_adj(joints_to_consider,temporal_kernel_size): # returns adj.matrix to be fed to the network
        # edgelist 32 notation
    edgelist_32 = [(0,1), (0,6), (0,12), (1,2), (2,3), (6,7),               (7,8), (12,13), (13,14), (13,17), (13,25),               (14,15), (17,18), (18,19), (25,26), (26,27)]
    # create edgelist 17 notation
    edgelist = []
    new_joints_idxs = list(range(0,joints_to_consider))
    if joints_to_consider==17:
        joints_to_keep=[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    else:
        joints_to_keep=range(0,joints_to_consider)

    #reindex joints from 32 notation - keys to 17 notation - values
    joints_idx_map = dict(zip(joints_to_keep,new_joints_idxs))
    for edge in edgelist_32:
        edgelist.append((joints_idx_map[edge[0]],joints_idx_map[edge[1]]))
    # create a graph
    G=nx.Graph()
    G.add_edges_from(edgelist)
    # create adjacency matrix
    A = nx.adjacency_matrix(G,nodelist=new_joints_idxs).todense()
    #normalize adjacency matrix
    A=normalize_A(A)
    return torch.Tensor(spatio_temporal_graph(joints_to_consider,temporal_kernel_size,A))


def mpjpe_error(batch_pred,batch_gt,normalize,train_stats,vis=False):
    #assert batch_pred.requires_grad==True
    #assert batch_gt.requires_grad==False

    '''batch_gt comes unnormalized, while batch_pred is normalized,
    to compute loss, campared to data without normalization,
    batch_pred should be unnormalized to original data'''
    if vis==True:
        batch_pred = Prediction_Unnormalization_Test(batch_pred,normalize,train_stats,unnorm=True)
    elif vis==False:
        batch_pred = Prediction_Unnormalization(batch_pred,normalize,train_stats,unnorm=True)
    
    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))

def Prediction_Unnormalization(input_tensor,normalize,train_stats,unnorm=True):
    if unnorm == True:
        if normalize == True:
            #Mean std
            data_mean_train = train_stats[0]
            data_std_train = train_stats[1]
            meanTensor = np.repeat(data_mean_train,input_tensor.shape[0],axis=0) # repeat along N axis
            stdTensor = np.repeat(data_std_train,input_tensor.shape[0],axis=0)
            frames = [meanTensor for _ in range(input_tensor.shape[1])] # repeat along frame axis = 10
            meanTensor = np.swapaxes(np.stack(frames),0,1) # change axes N and frames
            frames = [stdTensor for _ in range(input_tensor.shape[1])] # repeat along frame axis = 10
            stdTensor = np.swapaxes(np.stack(frames),0,1) # change axes N and frames
            UnnormalizedTensor = torch.mul(input_tensor, torch.Tensor(stdTensor).cuda()) + torch.Tensor(meanTensor).cuda()
        elif normalize == False:
            avg_len = train_stats[0]
            UnnormalizedTensor = input_tensor * avg_len
        elif normalize == None:
            UnnormalizedTensor = input_tensor
        return UnnormalizedTensor
    elif unnorm == False:
        # normalize prediction
        if normalize == True:
            #Mean std normalization
            data_mean_train = train_stats[0]
            data_std_train = train_stats[1]

            meanTensor = np.repeat(data_mean_train,input_tensor.shape[0],axis=0) # repeat along N axis
            stdTensor = np.repeat(data_std_train,input_tensor.shape[0],axis=0)
            frames = [meanTensor for _ in range(input_tensor.shape[1])] # repeat along frame axis = 10
            meanTensor = np.swapaxes(np.stack(frames),0,1) # change axes N and frames
            frames = [stdTensor for _ in range(input_tensor.shape[1])] # repeat along frame axis = 10
            stdTensor = np.swapaxes(np.stack(frames),0,1) # change axes N and frames
            
            normalizedTensor = torch.div(input_tensor - torch.Tensor(meanTensor).cuda(), torch.Tensor(stdTensor).cuda())
      
        elif normalize == False:
            # body len normalization
            avg_len = train_stats[0]
            normalizedTensor = input_tensor / avg_len
        elif normalize == None:
            normalizedTensor = input_tensor
        return normalizedTensor

def Prediction_Unnormalization_Test(input_tensor,normalize,train_stats,unnorm=True):
    if unnorm == True:
        if normalize == True:
            #Mean std
            data_mean_train = train_stats[0]
            data_std_train = train_stats[1]
            meanTensor = np.repeat(data_mean_train,input_tensor.shape[0],axis=0) # repeat along N axis
            stdTensor = np.repeat(data_std_train,input_tensor.shape[0],axis=0)
            meanTensor = meanTensor.float()
            stdTensor = stdTensor.float()
            input_tensor = torch.Tensor(input_tensor)
            input_tensor = input_tensor.float()
            UnnormalizedTensor = torch.mul(input_tensor, stdTensor.cpu()) + meanTensor.cpu()
        elif normalize == False:
            avg_len = train_stats[0]
            UnnormalizedTensor = input_tensor * avg_len
        elif normalize == None:
            UnnormalizedTensor = input_tensor
        return UnnormalizedTensor
    elif unnorm == False:
        # normalize prediction
        if normalize == True:
            #Mean std normalization
            data_mean_train = train_stats[0]
            data_std_train = train_stats[1]

            meanTensor = np.repeat(data_mean_train,input_tensor.shape[0],axis=0) # repeat along N axis
            stdTensor = np.repeat(data_std_train,input_tensor.shape[0],axis=0)

            meanTensor = meanTensor.float()
            stdTensor = stdTensor.float()
            input_tensor = torch.Tensor(input_tensor)
            input_tensor = input_tensor.float()

            normalizedTensor = torch.div((input_tensor - meanTensor.cpu()), stdTensor.cpu())
            #normalizedTensor = np.divide((input_tensor - meanTensor),stdTensor)
        elif normalize == False:
            # body len normalization
            avg_len = train_stats[0]
            normalizedTensor = input_tensor / avg_len
        elif normalize == None:
            normalizedTensor = input_tensor
        return normalizedTensor



def Mean_Std_Tensor_Normalize(input_tensor, len_of_sequences_input, len_of_sequences_output,\
                              data_mean_train = 0, data_std_train = 0, train=True):

    if train:
        # create data statistics
        data_mean = np.mean(input_tensor, axis=(0,1)) # input_tensor shape (N,10,17,3)
        data_mean = data_mean.reshape(1,data_mean.shape[0],data_mean.shape[1])
        data_std = np.std(input_tensor, axis=(0,1))
        data_std = data_std.reshape(1,data_std.shape[0],data_std.shape[1])
        # data_mean and data_std shape (1, 17, 3)
    elif train == False:
        # use train statistics on test data
        data_mean = data_mean_train
        data_std = data_std_train

    meanTensor = np.repeat(data_mean,input_tensor.shape[0],axis=0) # repeat along N axis
    stdTensor = np.repeat(data_std,input_tensor.shape[0],axis=0)

    frames = [meanTensor for _ in range(input_tensor.shape[1])] # repeat along frame axis = 10
    meanTensor = np.swapaxes(np.stack(frames),0,1) # change axes N and frames
    frames = [stdTensor for _ in range(input_tensor.shape[1])] # repeat along frame axis = 10
    stdTensor = np.swapaxes(np.stack(frames),0,1) # change axes N and frames

    normalizedTensor = np.divide((input_tensor - meanTensor),stdTensor)
    #print(normalizedTensor.shape)

    len_of_all_sequences = len_of_sequences_input + len_of_sequences_output
    sequences_observed = normalizedTensor[:,0:len_of_sequences_input,:,:]
    #sequences_predict_gt = normalizedTensor[:,len_of_sequences_input:len_of_all_sequences,:,:]
    # gt is not normalized
    sequences_predict_gt = input_tensor[:,len_of_sequences_input:len_of_all_sequences,:,:]

    
    if train:
        return torch.from_numpy(data_mean), torch.from_numpy(data_std),\
         TensorDataset(torch.Tensor(sequences_observed).permute(0,3,1,2),\
         torch.Tensor(sequences_predict_gt).permute(0,3,1,2))
    else:
        return TensorDataset(torch.Tensor(sequences_observed).permute(0,3,1,2),\
         torch.Tensor(sequences_predict_gt).permute(0,3,1,2))


def Avg_Body_Length(input_tensor):
    '''WORKS ONLY FOR 17 JOINTS'''
    joints_to_keep=[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    new_joints_idxs = list(range(0,17))
    #reindex joints from 32 notation - keys to 17 notation - values
    joints_idx_map = dict(zip(joints_to_keep,new_joints_idxs))
    # edgelist 32 notation
    edgelist_32 = [(0,1), (0,6), (0,12), (1,2), (2,3), (6,7), \
            (7,8), (12,13), (13,14), (13,17), (13,25), \
            (14,15), (17,18), (18,19), (25,26), (26,27)]
    # create edgelist 17 notation
    edgelist = []
    for edge in edgelist_32:
        edgelist.append((joints_idx_map[edge[0]],joints_idx_map[edge[1]]))

    avg_body_length = []
    for i in range(len(input_tensor)):
        # create a graph
        G=nx.Graph()
        for edge in edgelist:        
            edge_length = l2_norm(input_tensor[i][0][edge[0]],input_tensor[i][0][edge[1]])
            G.add_edge(edge[0], edge[1], length=edge_length)

        body_length = G.size(weight='length')
        avg_body_length.append(body_length)
        del G
        
    return np.average(avg_body_length)


def Avg_Body_Length_Tensor_Normalize(input_tensor, len_of_sequences_input, len_of_sequences_output, avg_len_train=0, train=True):
    if train:
        # compute avg body len statistics for training data
        avg_len = Avg_Body_Length(input_tensor)
    elif train == False:
        # use training avg body len statistics for test data
        avg_len = avg_len_train

    normalizedTensor = input_tensor / avg_len
    len_of_all_sequences = len_of_sequences_input + len_of_sequences_output
    sequences_observed = normalizedTensor[:,0:len_of_sequences_input,:,:]
    # gt is not normalized
    #sequences_predict_gt = normalizedTensor[:,len_of_sequences_input:len_of_all_sequences,:,:]
    sequences_predict_gt = input_tensor[:,len_of_sequences_input:len_of_all_sequences,:,:]

    if train:
        return avg_len, TensorDataset(torch.Tensor(sequences_observed).permute(0,3,1,2),\
        torch.Tensor(sequences_predict_gt).permute(0,3,1,2))
    elif train == False:
        return TensorDataset(torch.Tensor(sequences_observed).permute(0,3,1,2),\
        torch.Tensor(sequences_predict_gt).permute(0,3,1,2))

def l2_norm(positions1,positions2):
    assert positions1.shape==positions2.shape==(3,)
    dist=np.sqrt((positions1[0]-positions2[0])**2+(positions1[1]-positions2[1])**2+(positions1[2]-positions2[2])**2)
  #  print(np.round(dist,4))
    return np.round(dist,4)


def No_normalization(input_tensor,len_of_sequences_input, len_of_sequences_output):

    len_of_all_sequences = len_of_sequences_input + len_of_sequences_output
    sequences_observed = input_tensor[:,0:len_of_sequences_input,:,:]
    sequences_predict_gt = input_tensor[:,len_of_sequences_input:len_of_all_sequences,:,:]

    return TensorDataset(torch.Tensor(sequences_observed).permute(0,3,1,2),\
    torch.Tensor(sequences_predict_gt).permute(0,3,1,2))




def all_actions(allactions):
    if allactions=='all':
        allactions=["Walking", "Eating", "Smoking", "Discussion", "Directions",
                   "Greeting", "Phoning", "Posing", "Purchases", "Sitting",
                   "SittingDown", "Photo", "Waiting", "WalkDog",
                   "WalkTogether"]
    return allactions
    
    
    
    
    
    
    
    
    
    
