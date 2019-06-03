#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/05/31
author: lujie
"""

import torchvision as tv
import torch.utils.data.dataloader as Data

root_path = '/home/lujie/Documents/deep_learning/cs231n/cs231n_dataset/'

class ToyData(object):

    def __init__(self, args_dict = {}):


        self.batch_size = args_dict.get('batch_size', 32)
        self.workers    = args_dict.get('workers', 4)

        self.loader = None


    def _dataloder(self, train_flag = True, shuffle_flag = False):

        self.loader = Data.DataLoader(tv.datasets.CIFAR10(root_path+'CIFAR10_data', train = train_flag, \
                          transform = tv.transforms.Compose([
                              tv.transforms.ToTensor(),
                              tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                          ]), download = False),
                          batch_size=self.batch_size, shuffle=shuffle_flag, num_workers=self.workers)

        return self.loader
