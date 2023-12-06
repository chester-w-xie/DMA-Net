"""
-------------------------------File info-------------------------
% - File name: My_functions.py
% - Description:
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2021-04-24
%  Copyright (C) PRMI, South China university of technology; 2021
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import os
import sys
import importlib
import random
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler


class SGDmScheduleV2:
    # -
    def __init__(self, optimizer, lr_init):
        self.lr_init = lr_init
        self.epochs = 250  # -
        self.start = 50  # -
        self.warmup = 0  # -
        self.lr_warmup = lr_init / 1000  # -
        self.after_lr = lr_init / 1000  # -
        self.optimizer = optimizer
        self.lr = self.lr_init

    # -
    def step(self, epoch):
        if epoch < self.warmup:
            self.lr = self.lr_warmup + (self.lr_init - self.lr_warmup) * epoch / self.warmup
        elif epoch < self.start:
            self.lr = self.lr_init
        elif epoch >= self.epochs:
            self.lr = self.after_lr
        else:
            self.lr = self.lr_init - self.lr_init * (epoch - self.start) / (self.epochs - self.start)

        adjust_learning_rate(self.optimizer, self.lr)

    def get_lr(self):
        return [self.lr]


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


global_first_time = True


class MixUpLossCrossEntropyLoss(nn.Module):
    """Adapts the loss function to go with mixup."""

    def __init__(self, crit=None):
        super().__init__()
        if crit is None:
            self.crit = nn.CrossEntropyLoss(reduction="none")
        else:
            self.crit = crit

    def forward(self, output, target1, target2=None, lmpas=None):
        global global_first_time
        if target2 is None:
            return self.crit(output, target1).mean()
        if global_first_time:
            print("using mix up loss!! ", self.crit)
        global_first_time = False

        loss1, loss2 = self.crit(output, target1), self.crit(output, target2)
        return (loss1 * lmpas + loss2 * (1 - lmpas)).mean()


def my_mixup_function(data, alpha):
    rn_indices = torch.randperm(data.size(0))
    lambd = np.random.beta(alpha, alpha, data.size(0))
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd.astype(np.float32))

    return rn_indices, lam


# -
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def calculate_accuracy(target, predict, classes_num1, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num1)
    total = np.zeros(classes_num1)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num2):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num2, classes_num2))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def worker_init_fn(x):
    seed = (torch.initial_seed() + x * 1000) % 2 ** 31  # problem with nearly seeded randoms

    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    return


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


class DictAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, num):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] = self.sum.get(k, 0) + val[k] * num
            self.count[k] = self.count.get(k, 0) + num
            self.avg[k] = self.sum[k] / self.count[k]