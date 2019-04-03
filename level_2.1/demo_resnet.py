#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/26
author: lujie
"""

import time
import torch
import torchvision as tv
from utils.resnet_utils import *
import torch.utils.data.dataloader as Data
from IPython import embed


if __name__ == '__main__':


    train_data = tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = True, \
                     transform = tv.transforms.Compose([
                         tv.transforms.ToTensor(),
                         tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                     ]), download = False)

    test_data  = tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = False, \
                     transform = tv.transforms.Compose([
                         tv.transforms.ToTensor(),
                         tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                     ]), download = False)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=4)
    test_loader  = Data.DataLoader(dataset=test_data, batch_size=128, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(10).to(device)
    model = net_trainer(model, train_loader, test_loader, 20, 128)
    # net_infer(model, test_loader)
