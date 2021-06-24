from config import cfg
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn

class pre():
    
    def __init__(self, train_data_path, test_data_path, 
                 y_train_path, y_val_path, batch_size, aspect_ratio):
        
        train_data = pd.read_pickle(train_data_path)
        test_data = pd.read_pickle(test_data_path)
        
        print("train_data_shape:", train_data.shape)
        print("test_data_shape:", test_data.shape)
        
        y_train = pd.read_pickle(y_train_path)
        y_val = pd.read_pickle(y_val_path)
        
        print("y_train_shape:", y_train.sum())
        print("y_val_shape:", y_val.sum())
        
        #split train data with accident and noaccident data 
        train_acc_index = np.where(y_train == 1)[0]
        train_nonacc_index = np.where(y_train == 0)[0]

        train_acc_data = train_data[train_acc_index]
        y_train_acc = y_train[train_acc_index]

        train_nonacc_data = train_data[train_nonacc_index]
        y_train_nonacc = y_train[train_nonacc_index]
        
        #make numpy as Tensor
        train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(y_train))
        train_acc_dataset = TensorDataset(torch.from_numpy(train_acc_data),    torch.from_numpy(y_train_acc))
        train_nonacc_dataset = TensorDataset(torch.from_numpy(train_nonacc_data), torch.from_numpy(y_train_nonacc))

        test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(y_val))
        
        # Batch size for training (change depending on how much memory you have)
        batch_size = batch_size

        #aspect_ratio_with 0, 1 
        aspect_ratio = aspect_ratio
        acc_batch_size = int(batch_size * aspect_ratio)

        #train_data_loader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=acc_batch_size, shuffle=False, drop_last=False)
        self.train_acc_dataloader = torch.utils.data.DataLoader(train_acc_dataset,batch_size=acc_batch_size, shuffle=True, drop_last=True)
        self.train_nonacc_dataloader = torch.utils.data.DataLoader(train_nonacc_dataset,batch_size=batch_size, shuffle=True, drop_last=True)

        #test_data_loader
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False, drop_last=False)