import torch
import numpy as np
import os
import sys
#sys.path.append('./')
sys.path.append('./../')

import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)



def _load_collection_real(filename,cuda_option):

    print('#loaded collection#')
    device = torch.device('cuda:0' if cuda_option else 'cpu')

    if filename == 'CO2':
        Dataset = sio.loadmat('./datasets/real/CO2data.mat')
        x_train, x_test, y_train, y_test = Dataset['xtrain'], Dataset['xtest'],Dataset['ytrain'], Dataset['ytest']
        x_train, x_test, y_train, y_test = np.float64(x_train), np.float64(x_test) , np.float64(y_train), np.float64(y_test)
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_test).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_test).to(device)


    elif filename == 'airline':
        Dataset = sio.loadmat('./datasets/real/airlinedata.mat')
        x_train, x_test, y_train, y_test = Dataset['xtrain'], Dataset['xtest'],Dataset['ytrain'], Dataset['ytest']
        x_train, x_test, y_train, y_test = np.float64(x_train), np.float64(x_test) , np.float64(y_train), np.float64(y_test)
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_test).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_test).to(device)


    elif filename == 'audio':
        Dataset = sio.loadmat('./datasets/real/audiodata.mat')
        x_train, x_test, y_train, y_test = Dataset['xtrain'], Dataset['xtest'],Dataset['ytrain'], Dataset['ytest']
        x_train, x_test, y_train, y_test = np.float64(x_train), np.float64(x_test) , np.float64(y_train), np.float64(y_test)
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_test).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_test).to(device)
    
    
    
    elif filename == 'RQ10PER4':
        Dataset = sio.loadmat('./datasets/synthetic/RQ10PER4.mat')
        x_train,y_train = Dataset['xt'],Dataset['yt'],
        x_train,y_train = np.float64(x_train), np.float64(y_train),
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_train).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_train).to(device)
    else :
        raise ValueError('NOT found dataset')

        
        

def _load_collection_syn(filename,cuda_option):
    print('#loaded collection#')
    device = torch.device('cuda:0' if cuda_option else 'cpu')

    print(os.getcwd())
    if filename == 'RQ10PER4':
        Dataset = sio.loadmat('./datasets/synthetic/RQ10PER4.mat')
        x_train,y_train = Dataset['xt'],Dataset['yt'],
        x_train,y_train = np.float64(x_train), np.float64(y_train),
        autocorelate = np.float64(Dataset['autocorrelate'])
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_train).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_train).to(device),autocorelate.squeeze()


    elif filename == 'SM_Q5_equal':
        Dataset = sio.loadmat('./datasets/synthetic/SM_Q5_equal2.mat')
        x_train, x_test, y_train, y_test = Dataset['x_train'], Dataset['x_test'],Dataset['y_train'], Dataset['y_test']
        x_train, x_test, y_train, y_test = np.float64(x_train), np.float64(x_test) , np.float64(y_train), np.float64(y_test)
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_test).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_test).to(device)

    elif filename == 'SM_Q5_unequal':
        Dataset = sio.loadmat('./datasets/synthetic/SM_Q5_unequal2.mat')
        x_train, x_test, y_train, y_test = Dataset['x_train'], Dataset['x_test'],Dataset['y_train'], Dataset['y_test']
        x_train, x_test, y_train, y_test = np.float64(x_train), np.float64(x_test) , np.float64(y_train), np.float64(y_test)
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_test).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_test).to(device)


    elif filename in ['SM_Q5_exp1_unequal_v3','SM_Q2_exp1'] :
        #Dataset = sio.loadmat('./../datasets/synthetic/' + filename + '.mat')
        Dataset = sio.loadmat('./datasets/synthetic/' + filename + '.mat')        
        x_train, x_test, y_train, y_test , x_full, y_full = Dataset['x_train'], Dataset['x_test'],Dataset['y_train'], Dataset['y_test'],Dataset['x_full'], Dataset['y_full']
        x_train, x_test, y_train, y_test, x_full,y_full = np.float64(x_train), np.float64(x_test) , np.float64(y_train), np.float64(y_test), np.float64(x_full), np.float64(y_full)
        return torch.from_numpy(x_train).to(device),torch.from_numpy(x_test).to(device),torch.from_numpy(x_full).to(device),\
               torch.from_numpy(y_train).to(device),torch.from_numpy(y_test).to(device),torch.from_numpy(y_full).to(device),Dataset


    else :
        raise ValueError('NOT found dataset')





filepath_dir = './datasets/uci_datasets/'
def _load_collection_uci_data(filename,random_seed=1111, numtotal = 200000, cuda_option = None , normalize = True):
    device = torch.device('cuda:0' if cuda_option else 'cpu')
    data_dir = filepath_dir + filename + '/'
    x_full = np.load(data_dir + 'X.npy')
    y_full = np.load(data_dir + 'y.npy')
    if normalize :
        y_full = np.asarray(y_full).reshape(1,-1)
        y_mean = y_full.mean(axis = 1)
        y_std = y_full.std(axis = 1)       
        y_full = (y_full - y_mean)/(y_std+1e-8)
        y_full = y_full.squeeze()
    else:
        y_full = np.asarray(y_full).reshape(1,-1)
        y_mean = y_full.mean(axis = 1)
        y_full = (y_full - y_mean)
        y_full = y_full.squeeze()
        
        
    if len(y_full) > numtotal:
        x_full = x_full[:numtotal]
        y_full = y_full[:numtotal]
        
    x_train, x_valtest, y_train, y_valtest = train_test_split(x_full, y_full, test_size = 0.2, random_state = random_seed)    
    x_val = x_valtest[::2] 
    y_val = y_valtest[::2]
    x_test = x_valtest[1::2] 
    y_test = y_valtest[1::2]
    return torch.from_numpy(x_train).to(device), torch.from_numpy(x_val).to(device), torch.from_numpy(x_test).to(device), \
           torch.from_numpy(y_train.reshape(-1,1)).to(device), torch.from_numpy(y_val.reshape(-1,1)).to(device), torch.from_numpy(y_test.reshape(-1,1)).to(device)
    


    
    
def _load_collection_uci_data_v2(filename,random_seed=1, numtotal = 200000, cuda_option = None , normalize = False ,numrep = 1):
    data_dir = filepath_dir  + filename + '/'
    x_full = np.load(data_dir + 'X.npy')
    y_full = np.load(data_dir + 'y.npy')    
    

    if normalize :
        print('normalized')        
        y_full = np.asarray(y_full).reshape(1,-1)
        y_mean = y_full.mean(axis = 1)
        y_std = y_full.std(axis = 1)       
        y_full = (y_full - y_mean)/(y_std+1e-8)
        y_full = y_full.squeeze()
    else:
        print('unnormalized')
        y_full = np.asarray(y_full).reshape(1,-1)
        y_mean = y_full.mean(axis = 1)
        y_full = (y_full - y_mean)
        y_full = y_full.squeeze()        
        
       
    x_shuffled, y_shuffled, shuffled_indices = shuffle(x_full, y_full, range(len(y_full)), random_state=random_seed)
    partition_length = int(len(x_shuffled)/numrep)
    x_shuffled_list = []
    y_shuffled_list = []
    for i in range(1,numrep+1):
        x_shuffled_list.append(x_shuffled[(i-1)*partition_length: i*partition_length])
        y_shuffled_list.append(y_shuffled[(i-1)*partition_length: i*partition_length])

    return np.asarray(x_shuffled_list),np.asarray(y_shuffled_list)






def _traintestsplit(ith_x_shuffled,ith_y_shuffled,device,random_seed):
    device = torch.device('cuda:0' if device else 'cpu')
    x_train, x_valtest, y_train, y_valtest = train_test_split(np.asarray(ith_x_shuffled),
                                                              np.asarray(ith_y_shuffled).reshape(-1,1),
                                                              test_size = 0.2,
                                                              random_state = random_seed)

    x_val = x_valtest[::2] 
    y_val = y_valtest[::2]
    x_test = x_valtest[1::2] 
    y_test = y_valtest[1::2]
    return torch.from_numpy(x_train).to(device), torch.from_numpy(x_val).to(device), torch.from_numpy(x_test).to(device), \
           torch.from_numpy(y_train.reshape(-1,1)).to(device), torch.from_numpy(y_val.reshape(-1,1)).to(device), torch.from_numpy(y_test.reshape(-1,1)).to(device)
    









if __name__ == "__main__":
    path_filename = 'RQ10PER4'
    #path_filename = 'airline'
    #x_train,y_train = _load_collection_data(path_filename,cuda_option=True)
    #print(x_train,y_train)