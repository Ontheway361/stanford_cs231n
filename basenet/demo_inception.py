#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/05/31
author: lujie
"""

import time
import torch
import torchvision as tv
from utils.resnet_utils import net_trainer
import torch.utils.data.dataloader as Data
from utils.inception_utils import GoogLeNet

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

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = GoogLeNet(10).to(device)
    model = GoogLeNet(10)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    net_trainer(model, train_loader, test_loader, 20, 128)
    # net_infer(model, test_loader)
