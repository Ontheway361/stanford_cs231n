#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
date  : 2019/01/17
author: lujie
"""

import torch as t
from tqdm import tqdm
from torch.autograd import Variable


class LeNets(t.nn.Module):

    def __init__(self, num_class = 10):
        super(LeNets, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(3, 64, 7),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 5),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2)
        )
        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, 3),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2)
        )
        self.dense = t.nn.Sequential(
            t.nn.Linear(256, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, num_class)
        )

    def forward(self, x):

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)

        return out
