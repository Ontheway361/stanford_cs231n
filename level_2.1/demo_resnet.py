#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/26
author: lujie
"""

import time
import torch as t
import torchvision as tv
from utils.resnet_utils import *
import torch.utils.data.dataloader as Data
from IPython import embed


if __name__ == '__main__':


    train_data = tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = True, \
                     transform = tv.transforms.ToTensor(), download=False)

    test_data  = tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = False, \
                     transform = tv.transforms.ToTensor(), download=False)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    test_loader  = Data.DataLoader(dataset=test_data, batch_size=128)

    model = ResNet(10)
    model = net_trainer(model, train_loader, 10, 128)
    net_test(model, test_loader)
