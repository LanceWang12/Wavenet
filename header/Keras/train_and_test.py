import time
import pandas as pd
import numpy as np
import tensorflow
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from header.Keras.dataset import generate_data

def scheduler(epoch, lr):
    # step = 10
    # if epoch % step == 0:
    #     return lr * tensorflow.math.exp(-0.1)
    # else:
    #     return lr
    return lr * tensorflow.math.exp(-0.1)

def train_val_plot(history, metric):
    train_loss = history.history['loss']
    train_metric = history.history[metric]

    val_loss = history.history['val_loss']
    val_metric = history.history['val_' + metric]

    # Learning rate
    plt.figure(figsize = (8, 6))
    plt.title('learning rate history')
    plt.plot(history.history['lr'], label = 'lr')
    plt.legend()
    plt.show()
    
    # Loss
    plt.figure(figsize = (8, 6))
    plt.title('loss')
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.legend()
    plt.show()

    # Metric
    plt.figure(figsize = (10, 6))
    plt.title(metric)
    plt.plot(train_metric, label = 'train_{}'.format(metric))
    plt.plot(val_metric, label = 'val_{}'.format(metric))
    plt.legend()
    plt.show()
    
def rolling_testing(model, model_tag, data, data_info, baseline, rolling_info):
    # Parameter
    # model: 預測是之模型
    # model_tag: 要測試的是 deep learning or machine learning('dl' or 'ml')
    # data: 經過特徵工程的資料集

    # -----------------------------------------------
    # data_info: 為一字典，存放資料如下
    # data_info = {
    #     'label': dataframe 中 label 的欄位名稱, 
    #     'date': dataframe 中 target_date 的欄位名稱,
    #     'start': 測試開始日期,
    #     'end': 測試結束日期
    # }
    # -----------------------------------------------

    # -----------------------------------------------
    # rolling_info: 為一字典，存放資料如下
    # rolling_info = {
    #     'lag': 向前看幾天,
    #     'gap': 預測未來第幾天,
    #     'check_period': 檢查週期,
    #     'check_size': 計算準確率的資料數量,
    #     'sample': 是否使用 sample 的方式,
    #     'optimizer': Keras 的優化器(Ex. Adam),
    #     'metric': Keras 的評價方式(Ex. binary_accuracy),
    #     'scheduler': Scheduler,
    #     'early_stop': Early_stopping
    # }
    # -----------------------------------------------

    lag, gap = rolling_info['lag'], rolling_info['gap']
    check_period, check_size = rolling_info['check_period'], rolling_info['check_size']
    sample = rolling_info['sample']

    if model_tag == 'dl':
        optimizer = rolling_info['optimizer']
        metric = rolling_info['metric']
        scheduler = rolling_info['scheduler']
        early_stop = rolling_info['early_stop']
        
        X, Y, date = generate_data(df = data, label_idx = data_info['label'], model_tag = 'dl',
                                date_idx = data_info['date'], lag = lag, gap = gap,
                                normalize_method = 'min-max',
                                start = None, end = None)
    elif model_tag == 'ml':
        X, Y, date = generate_data(df = data, label_idx = data_info['label'], model_tag = 'ml',
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
        
    print("Roll the model from {} to {}.".format(date[test_start], date[test_end]))
    start = time.time()
    if sample:
        sample_size = int(check_size * 2 / 3)
        prob = gen_chi2_prob(n = check_size)
    sample_weight = gen_chi2_prob(n = 3000)
    acc_lst = []
    pred_result = []
    cnt = 0
    for i in range(test_start, test_end):
        x_tmp = X[i][np.newaxis, :]
        pred = np.round(model.predict(x_tmp).reshape(-1))
        pred_result.append(pred[0])

        # 滿足檢查條件
        if (cnt > gap) and ((cnt - (gap + check_period - 1)) % check_period == 0):
            # 是否要用 sample 方式評估
            if sample:
                sample_idx = np.random.choice(np.arange((i - gap - check_size), (i - gap)),
                                              size = (sample_size, ),
                                              replace = False, p = prob)
                x_tmp = X[sample_idx]
                y_ma = baseline[sample_idx]
                y_true = Y[sample_idx]
            else:
                x_tmp = X[(i - gap - check_size): (i - gap)]
                y_ma = baseline[(i - gap - check_size): (i - gap)]
                y_true = Y[(i - gap - check_size): (i - gap)]
            
            
            # Is deep learning or machine learning?
            if model_tag == 'dl':
                y_pred = np.round(model.predict(x_tmp).reshape(-1))
            elif model_tag == 'ml':
                y_pred = np.round(model.predict(x_tmp))
            else:
                raise NotImplementedError("I don't know this model_tag!!")
            
            acc = accuracy_score(y_true, y_pred) * 100
            ma_acc = accuracy_score(y_true, y_ma) * 100
            acc_lst.append(acc)
            
            
            if acc + 5 < ma_acc:
                print('Day {} start to retrain...'.format(date[i - gap]))
                print('Ma, model = {:.2f}, {:.2f}'.format(ma_acc, acc))
                
                if model_tag == 'dl':
                    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = metric)
                    model.fit(x = X[(i - gap-3000): (i - gap)], y = Y[(i - gap-3000): (i - gap)], batch_size = 256, epochs=20, 
                            verbose = 1, callbacks = [scheduler, early_stop], validation_split=0, 
                            validation_data=None, shuffle=True, class_weight=None,
                            sample_weight=sample_weight, initial_epoch=0, steps_per_epoch=None,
                            validation_steps=None)
                else:
                    model.fit(X[(i - gap-3000): (i - gap)], Y[(i - gap-3000): (i - gap)],
                              sample_weight=sample_weight)
                
                
            
        cnt += 1
    end = time.time()
    print('End in {:.2f}s.'.format(end - start))
    acc_lst = np.array(acc_lst)
    pred_result = np.array(pred_result)
    return acc_lst, pred_result


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


def Cal_rolling_acc(predict, true, start, end, lag, gap):
    pass