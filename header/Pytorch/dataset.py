import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, date
from scipy.stats import mode

class MyDataset(Dataset):
    def __init__(self, df, label_idx, date_idx, lag = 22, gap = 22, normalize_method = 'min-max',
                 start = None, end = None):

        self.x, self.y, self.tx_dt, self.target_dt = generate_data(df = df, label_idx = label_idx, date_idx = date_idx, 
                                                                   model_tag = 'dl', lag = lag, 
                                                                   gap = gap, normalize_method = 'min-max',
                                                                   start = start, end = end)
        print(self.x.shape, self.y.shape)
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

class Copy(Dataset):
    def __init__(self, val):
        self.x, self.y = val
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

def generate_data(df, label_idx, date_idx, model_tag = 'dl', lag = 28, gap = 22, normalize_method = 'min-max', start = None, end = None):
    # date_arr store date (ex. ['2018-01-01', '2018-01-02', ...])
    date_arr = np.array(df[date_idx])
    Y = np.array(df[label_idx])
    X = np.array(df.drop(columns = [date_idx, label_idx]))
    
    if model_tag == 'ml':
        return X, U, date_arr

    # transform the start and end from date to index
    if (start != None) and (start not in date_arr):
        raise RuntimeError("Start meet Saturday or Sunday!! Please choose other date.")
    if (end != None) and (end not in date_arr):
        raise RuntimeError("End meet Saturday or Sunday!! Please choose other date.")

    if start == None:
        start = 0
    else:
        start = np.where(date_arr == start)[0][0]
    
    if end == None:
        end = X.shape[0]
    else:
        end = np.where(date_arr == end)[0][0]

    X, date_arr, Y = X[start: end], date_arr[start: end], Y[start: end]
    
    if np.isnan(np.min(X)):
        raise RuntimeError("Before normalization, there is some Nan in your dataset!! Please remove.")

    if normalize_method.lower() == 'min-max':
        Max = np.max(X, axis = 0)
        Min = np.mean(X, axis = 0)
        X = (X - Min) / (Max - Min + 1e-8)
    elif normalize_method.lower() == 'standarization':
        mean = np.mean(X, axis = 0)
        std = np.std(X, axis = 0)
        X = (X - mean) / (std + 1e-8)
    else:
        raise NotImplementedError("I don't know that normalize method!!")
    
    if np.isnan(np.min(X)):
        raise RuntimeError("After normalization, there is some Nan in your dataset!! Please remove.")

    # Generate data with label
    data_lst = []
    label_lst = []
    target_dt = []
    tx_dt = []

    for i in range(0, X.shape[0], 1):
        if (i + lag + gap) < X.shape[0]:
            data_lst.append(X[i: i + lag])
            label_lst.append(Y[i + lag])
            tx_dt.append(date_arr[i + lag])
            target_dt.append(date_arr[i + lag + gap])
    X = np.array(data_lst)
    Y = np.array(label_lst)
    tx_dt = np.array(tx_dt)
    target_dt = np.array(target_dt)

    # tuning the shape to fit model's input
    X = X.reshape((-1, X.shape[2], lag))
    Y = Y.reshape((-1, 1))

    return X, Y, tx_dt, target_dt

