"""
-------------------------------File info-------------------------
% - File name: Network_define.py
% - Description:
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2021-05-06
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
import torch.nn.functional as F
import math
import numpy as np

from torchsummary import summary


layer_index_total = 0
first_RUN = True

# Mutual attention module, MAM
class MAM(nn.Module):
    def __init__(self, c_in):
        super(MAM, self).__init__()

        self.query_a = nn.Conv2d(c_in, c_in, (1, 1))
        self.query_b = nn.Conv2d(c_in, c_in, (1, 1))

        self.key_a = nn.Conv2d(c_in, c_in, (1, 1))
        self.key_b = nn.Conv2d(c_in, c_in, (1, 1))

        self.value_a = nn.Conv2d(c_in, c_in, (1, 1))
        self.value_b = nn.Conv2d(c_in, c_in, (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))

        self.res_block_a1 = BasicBlock(in_channels=c_in*2, out_channels=c_in*2, k1=1, k2=1, stride=1)
        self.res_block_b1 = BasicBlock(in_channels=c_in*2, out_channels=c_in*2, k1=1, k2=1, stride=1)

        self.res_block_a2 = BasicBlock(in_channels=c_in*2, out_channels=c_in*4, k1=1, k2=1, stride=1)
        self.res_block_b2 = BasicBlock(in_channels=c_in*2, out_channels=c_in*4, k1=1, k2=1, stride=1)

        self.out_conv_a = nn.Sequential(nn.Conv2d(c_in*4, c_in, (1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=False),
                                        nn.BatchNorm2d(c_in))
        self.out_conv_b = nn.Sequential(nn.Conv2d(c_in*4, c_in, (1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=False),
                                        nn.BatchNorm2d(c_in))

    def forward(self, x_a, x_b):
        b, c, w_a, h_a = x_a.size()
        _, _, w_b, h_b = x_b.size()

        q_a = self.query_a(x_a).view(b, c, -1).transpose(1, 2)    # [B,(W_a*H_a),C_a]
        q_b = self.query_b(x_b).view(b, c, -1)  # [B, C_b, (W_b*H_b)]

        k_a = self.key_a(x_a).view(b, c, -1)  # [B,C_a,(W_a*H_a)]
        k_b = self.key_b(x_b).view(b, c, -1)  # [B, C_b, (W_b*H_b)]

        energy = torch.bmm(q_a, q_b)  # [B, (W_a*H_a), (W_b*H_b)]

        energy = energy.view(b, -1)  # [B, (W_a*H_a)*(W_b*H_b)]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        attention = attention.view(b, (w_a*h_a), (w_b*h_b))

        x_b_attention = torch.bmm(attention, k_b.transpose(1, 2))
        x_b_attention = x_b_attention.transpose(1, 2)  # [B, C_b,(W_a*H_a)]
        y_a = torch.cat([x_a, x_b_attention.view(b, c, w_a, h_a)], dim=1)

        x_a_attention = torch.bmm(attention.transpose(1, 2), k_a.transpose(1, 2))
        # [B,(W_b*H_b), C_a]
        x_a_attention = x_a_attention.transpose(1, 2)  # [B, C_a, (W_b*H_b)]

        y_b = torch.cat([x_b, x_a_attention.view(b, c, w_b, h_b)], dim=1)

        y_a = self.res_block_a1(y_a)
        y_a = self.res_block_a2(y_a)
        y_b = self.res_block_b1(y_b)
        y_b = self.res_block_b2(y_b)

        y_a = self.out_conv_a(y_a)
        y_b = self.out_conv_b(y_b)
        return y_a, y_b


class DMA_Net(nn.Module):

    def __init__(self):
        super(DMA_Net, self).__init__()

        n_classes = 10
        self.pooling_padding = 0
        self.lamda = nn.Parameter(torch.zeros(10))
        block = BasicBlock

        self.in_c1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.stage_A = self._make_stage(in_channels=128, out_channels=128, n_blocks=4, block=block, stride=1,
                                         maxpool=[1, 2, 4], k1s=[3, 3, 3, 3], k2s=[1, 3, 3, 1])

        self.in_c2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.stage_B = self._make_stage(in_channels=128, out_channels=128, n_blocks=4, block=block, stride=1,
                                         maxpool=[1, 2, 4], k1s=[3, 3, 3, 3], k2s=[1, 3, 3, 1])

        self.MAM = MAM(c_in=128)

        self.feed_forward_A = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=(1, 1),
                      stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(n_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.feed_forward_B = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=(1, 1),
                      stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(n_classes),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # initialize weights
        self.apply(initialize_weights)
        if isinstance(self.feed_forward_A[0], nn.Conv2d):
            self.feed_forward_A[0].weight.data.zero_()
            self.feed_forward_B[0].weight.data.zero_()
        self.apply(initialize_weights_fixup)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, maxpool=None, k1s=None,
                    k2s=None):
        stage = nn.Sequential()
        if 0 in maxpool:
            stage.add_module("maxpool{}_{}".format(0, 0)
                             , nn.MaxPool2d(2, 2, padding=self.pooling_padding))
        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1),
                             block(in_channels,
                                   out_channels,
                                   stride=stride, k1=k1s[index], k2=k2s[index]))

            in_channels = out_channels
            stride = 1
            # if index + 1 in maxpool:
            for m_i, mp_pos in enumerate(maxpool):
                if index + 1 == mp_pos:
                    stage.add_module("maxpool{}_{}".format(index + 1, m_i)
                                     , nn.MaxPool2d(2, 2, padding=self.pooling_padding))
        return stage

    def DPLM_A(self, x):
        # Deep representation learning module, DPLM_A
        global first_RUN

        if first_RUN:
            print("x:", x.size())
        x = self.in_c1(x)
        if first_RUN:
            print("in_c1:", x.size())

        x = self.stage_A(x)

        if first_RUN:
            print("stage_a1:", x.size())

        return x

    def DPLM_B(self, x):
        #  Deep representation learning module, DPLM_B
        global first_RUN

        if first_RUN:
            print("x:", x.size())
        x = self.in_c2(x)
        if first_RUN:
            print("in_c2:", x.size())

        x = self.stage_B(x)

        if first_RUN:
            print("stage_b1:", x.size())

        return x

    def forward(self, x1, x2):
        global first_RUN

        x1 = self.DPLM_A(x1)
        x2 = self.DPLM_B(x2)
        y1, y2 = self.MAM(x1, x2)

        y1 = self.feed_forward_A(y1)
        y2 = self.feed_forward_B(y2)

        # - Weighted Sum
        y1 = y1.view(y1.shape[0:2])
        y2 = y2.view(y2.shape[0:2])
        logit = torch.sigmoid(self.lamda)*y1 + (1-torch.sigmoid(self.lamda))*y2


        if first_RUN:
            print("logit:", logit.size())
        first_RUN = False
        return logit


def calc_padding(kernal):
    try:
        return kernal // 3
    except TypeError:
        return [k // 3 for k in kernal]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3):
        super(BasicBlock, self).__init__()
        global layer_index_total
        self.layer_index = layer_index_total
        layer_index_total = layer_index_total + 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(k1, k1), stride=stride,  # downsample with first conv
                               padding=calc_padding(k1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(k2, k2), stride=(1, 1), padding=calc_padding(k2),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=(1, 1), stride=stride,  # downsample
                          padding=(0, 0),
                          bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


def initialize_weights_fixup(module):
    if isinstance(module, BasicBlock):

        b = module
        n = b.conv1.kernel_size[0] * b.conv1.kernel_size[1] * b.conv1.out_channels
        b.conv1.weight.data.normal_(0, (layer_index_total ** (-0.5)) * math.sqrt(2. / n))
        b.conv2.weight.data.zero_()
        if b.shortcut._modules.get('conv') is not None:
            convShortcut = b.shortcut._modules.get('conv')
            n = convShortcut.kernel_size[0] * convShortcut.kernel_size[1] * convShortcut.out_channels
            convShortcut.weight.data.normal_(0, math.sqrt(2. / n))
    if isinstance(module, nn.Conv2d):
        pass

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


if __name__ == '__main__':
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DMA_Net()

    model.to(device)
    # summary(model, [(2, 256, 431), (2, 1728, 1581)])
    summary(model, [(2, 256, 431), (2, 256, 213)])

