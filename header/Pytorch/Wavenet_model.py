import time, copy
from collections import OrderedDict
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from header.Pytorch.Wavenet_unit import *

# Regression model
class Wavenet(nn.Module):
    def __init__(self, time_step, feature_num, num_blocks,
                 num_layers, output_channel, kernel_size):
        # Check kernek size
        # if kernel_size % 2 == 0:
        #     raise NotImplementedError("kernel_size can't be even number!!")

        super(Wavenet, self).__init__()
        self.time_step = time_step
        self.feature_num   = feature_num 
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.receptive_field = 1 + (kernel_size - 1) * \
                               num_blocks * sum([2**k for k in range(num_layers)])
        self.output_width = time_step - self.receptive_field + 1
        print('receptive_field: {}'.format(self.receptive_field))
        print('Output width: {}\n'.format(self.output_width))
        
        self.set_device()

        hs = []
        batch_norms = []

        # add gated convs
        first = True
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                if first:
                    h = GatedResidualBlock(feature_num ,  output_channel, kernel_size, 
                                           self.output_width, dilation=rate)
                    first = False
                else:
                    h = GatedResidualBlock(output_channel, output_channel, kernel_size,
                                           self.output_width, dilation=rate)
                h.name = 'b{}-l{}'.format(b, i)

                hs.append(h)
                batch_norms.append(nn.BatchNorm1d(output_channel))

        self.hs = nn.ModuleList(hs)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.relu = nn.ReLU()
        self.conv_1_1 = nn.Conv1d(output_channel, output_channel, 1)
        self.conv_1_2 = nn.Conv1d(output_channel, 1, 2)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(15, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for layer, batch_norm in zip(self.hs, self.batch_norms):
            x, skip = layer(x)
            x = batch_norm(x)
            skips.append(skip)

        x = reduce((lambda a, b : torch.add(a, b)), skips)
        x = self.relu(self.conv_1_1(x))
        x = self.dropout(x)
        x = self.relu(self.conv_1_2(x))
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sig(x).view(-1, 1)
        return x

    def predict(x):
        return self.forward(x)
    
    def flatten_size(self, x):
        shape = x.shape
        mul = 1
        for i in shape:
            mul *= i
        return mul

    def set_device(self, device=None):
        if device is None:
            self.device = torch.device('cuda:1')
        else:
            self.device = device

    def Train(self, dataloader, loss_func, optimizer, scheduler, Val = None, num_epochs = 25, disp_interval = 1):
        start = time.time()
        
        self.criterion = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.to(self.device)

        if Val != None:
            x_test, y_test = Val[:]
            x_test = x_test.float().to(self.device)
            y_test = y_test.to(self.device)

            val_info = {'loss': [], 'acc': []}
            self.max_val_acc = 0

        train_info = {'loss': [], 'acc': []}

        print('Start to train......')

        for epoch in range(1, num_epochs + 1):
            self.scheduler.step()
                
            # reset loss for current phase and epoch
            running_loss = 0.0
            running_acc = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.train()
                self.optimizer.zero_grad()
                
                # track history only during training phase
                outputs = self(inputs)
                loss = self.criterion(outputs, labels) 
                loss.backward()
                self.optimizer.step()
                        
                running_loss += loss.item()
                y_true = labels.to('cpu').numpy()
                pred = np.round(outputs.to('cpu').detach().numpy().reshape(-1))
                running_acc += accuracy_score(y_true, pred)

            # losses.append(running_loss)
            if epoch % disp_interval == 0:
                # Print training information
                epoch_loss, epoch_acc = running_loss / len(dataloader), running_acc / len(dataloader)
                train_info['loss'].append(epoch_loss)
                train_info['acc'].append(epoch_acc)
                print('Epoch {} / {}'.format(epoch, num_epochs))
                print('Learning Rate: {}'.format(self.scheduler.get_lr()))
                print('Training Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))
                
                if Ｖal != None:
                    # Print validation information
                    self.eval()
                    val_pred = self(x_test)
                    val_loss = self.criterion(val_pred, y_test).item()
                    y_true = y_test.to('cpu').numpy().reshape(-1)
                    val_pred = np.round(val_pred.to('cpu').detach().numpy().reshape(-1))
                    val_acc = accuracy_score(y_true, val_pred)
                    if self.max_val_acc < (val_acc + epoch_acc) / 2:
                        self.max_val_acc = (val_acc + epoch_acc) / 2
                        torch.save(self, './Pytorch_model_ckpt/Wavenet.pkl')

                    val_info['loss'].append(val_loss)
                    val_info['acc'].append(val_acc)

                    print('Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))

                print('-' * 30)
                print()
        end = time.time()
        print('Max validation accuracy: {:.4f}'.format(self.max_val_acc))
        print('\nEnd in {:.2f}s.'.format(end - start))
        return train_info, val_info


def _flatten(t):
    t = t.to(torch.device('cpu'))
    return t.data.numpy().reshape([-1])

