#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/26
author: lujie
"""

import torch as t
import numpy as np
from tqdm import tqdm
from torch import  nn
from IPython import embed
from torch.autograd import Variable
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    ''' Residual Block '''

    def __init__(self, inchannel, outchannel, stride = 1, shortcut = None):

        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):

        out = self.left(x)
        residual = None
        if self.right is None:
            residual = x
        else:
            residual = self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    ''' ResNet34 '''

    def __init__(self, num_classes=1000):

        super(ResNet, self).__init__()

        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # repeat layers with 3，4，6，3 residual blocks
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # fc
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self,  inchannel, outchannel, block_num, stride = 1):
        ''' construct the module pattern '''

        shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        ''' forward of resnet34 '''

        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 1)  # 7
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        
        return out
