import time
import pandas as pd
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from header.Pytorch.dataset import MyDataset, Copy
from torch.utils.data import DataLoader


def rolling_testing(model, model_tag, data, data_info, baseline, rolling_info):
    # Parameter
    # start: testing 開始日期
    # end: testing 結束日期

    device = torch.device('cuda:1')

    lag, gap = rolling_info['lag'], rolling_info['gap']
    check_period, check_size = rolling_info['check_period'], rolling_info['check_size']
    sample = rolling_info['sample']

    if model_tag == 'dl':
        optimizer = rolling_info['optimizer']
        scheduler = rolling_info['scheduler']
        loss_func = rolling_info['loss_func']
        
        Data = MyDataset(df = data, label_idx = data_info['label'], date_idx = data_info['date'], lag = lag, gap = gap,
                               normalize_method = 'min-max', start = None, end = None)
        X, Y = Data[:]
        date = Data.date_tick
    elif model_tag == 'ml':
        pass
    
    
    # transform the start and end from date to index
    if (data_info['start'] != None) and (data_info['start'] not in date):
        raise RuntimeError("Start meet Saturday or Sunday!! Please choose other date.")
    if (data_info['end'] != None) and (data_info['end'] not in date):
        raise RuntimeError("End meet Saturday or Sunday!! Please choose other date.")

    if data_info['start'] == None:
        test_start = 0
    else:
        test_start = np.where(date == data_info['start'])[0][0]
    
    if data_info['end'] == None:
        test_end = X.shape[0] - 1
    else:
        test_end = np.where(date == data_info['end'])[0][0]
        
    print("Roll the model from {} to {}.".format(date[test_start], date[test_end]))
    start = time.time()
    if sample:
        sample_size = int(check_size * 2 / 3)
        prob = gen_chi2_prob(n = check_size)
    sample_weight = gen_chi2_prob(n = 3000)
    acc_lst = []
    cnt = 0
    for i in range(test_start, test_end):
        # 滿足檢查條件
        if (cnt > gap) and ((cnt - (gap + check_period - 1)) % check_period == 0):
            # 是否要用 sample 方式評估
            if sample:
                sample_idx = np.random.choice(np.arange((i - gap - check_size), (i - gap)),
                                              size = (sample_size, ),
                                              replace = False, p = prob)
                x_tmp = X[sample_idx].to(device)
                y_ma = baseline[sample_idx]
                y_true = Y[sample_idx].detach().numpy().reshape(-1)
            else:
                x_tmp = X[(i - gap - check_size): (i - gap)].to(device)
                y_ma = baseline[(i - gap - check_size): (i - gap)]
                y_true = Y[(i - gap - check_size): (i - gap)].detach().numpy().reshape(-1)
            
            
            # Is deep learning or machine learning?
            if model_tag == 'dl':
                y_pred = np.round(model(x_tmp).to('cpu').detach().numpy().reshape(-1))
            elif model_tag == 'ml':
                y_pred = np.round(model.predict(x_tmp))
            else:
                raise NotImplementedError("I don't know this model_tag!!")
            
            acc = accuracy_score(y_true, y_pred) * 100
            ma_acc = accuracy_score(y_true, y_ma) * 100
            acc_lst.append(acc)
            
            
            if acc + 5 < ma_acc:
                print('\nDay {} start to retrain...\n'.format(date[i - gap]))
                print('Ma, model = {:.2f}, {:.2f}'.format(ma_acc, acc))
                
                if model_tag == 'dl':
                    Train = Copy(Data[(i - gap-3000): (i - gap)])
                    train_idx = int(4 / 5 * len(Train))
                    Val = Copy(Train[train_idx: ])
                    Train = Copy(Train[: train_idx])
                    train_loader = DataLoader(dataset=Train, batch_size=128, 
                                              shuffle=True, num_workers=8)
                    
                    model.Train(dataloader = train_loader, loss_func = loss_func,
                                optimizer = optimizer, scheduler = scheduler,
                                Val = Val, num_epochs = 20, disp_interval = 1)
                else:
                    pass
                
                
            
        cnt += 1
    end = time.time()
    print('End in {:.2f}s.'.format(end - start))
    acc_lst = np.array(acc_lst)
    return acc_lst

def gen_chi2_prob(n, field = (0, 10), k = 2):
    x = np.linspace(field[0], field[1], n)
    out = chi2.pdf(x, k)
    out = out / out.sum()
    return out[::-1]

def testing(model, model_tag, data, data_info, rolling_info):
    # Parameter
    # start: testing 開始日期
    # end: testing 結束日期
    lag, gap = rolling_info['lag'], rolling_info['gap']
    check_period, check_size = rolling_info['check_period'], rolling_info['check_size']
    sample = rolling_info['sample']

    X, Y, date = generate_data(df = data, label_idx = data_info['label'],
                               date_idx = data_info['date'], lag = lag, gap = gap,
                               normalize_method = 'min-max',
                               start = None, end = None)
    
    
    # transform the start and end from date to index
    if (data_info['start'] != None) and (data_info['start'] not in date):
        raise RuntimeError("Start meet Saturday or Sunday!! Please choose other date.")
    if (data_info['end'] != None) and (data_info['end'] not in date):
        raise RuntimeError("End meet Saturday or Sunday!! Please choose other date.")

    if data_info['start'] == None:
        test_start = 0
    else:
        test_start = np.where(date == data_info['start'])[0][0]
    
    if data_info['end'] == None:
        test_end = X.shape[0] - 1
    else:
        test_end = np.where(date == data_info['end'])[0][0]
        
    print("Test the model from {} to {}.".format(date[test_start], date[test_end]))
    
    if sample:
        sample_size = int(check_size * 2 / 3)
        prob = gen_chi2_prob(n = check_size)

    acc_lst = []
    for i in range(test_start + 5, test_end, 5):
        if sample:
            sample_idx = np.random.choice(np.arange((i - gap - check_size), (i - gap)),
                                        size = (sample_size, ),
                                        replace = False, p = prob)
            x_tmp = X[sample_idx]
            y_true = Y[sample_idx]
        else:
            x_tmp = X[(i - gap - check_size): (i - gap)]
            y_true = Y[(i - gap - check_size): (i - gap)]

        y_pred = np.round(model.predict(x_tmp).reshape(-1))
        acc = accuracy_score(y_true, y_pred) * 100
        acc_lst.append(acc)
    acc_lst = np.array(acc_lst)
    return acc_lst