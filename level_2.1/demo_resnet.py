#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/26
author: lujie
"""

import torch as t
# from utils.resnet_utils import *
from IPython import embed


if __name__ == '__main__':


    mnist_train = dset.MNIST('../../cs231n_dataset/MNIST_data', train=True, download=True,
                               transform=T.ToTensor())
    loader_train = DataLoader(mnist_train, batch_size=batch_size,
                              sampler=ChunkSampler(NUM_TRAIN, 0))

    mnist_val = dset.MNIST('../../cs231n_dataset/MNIST_data', train=True, download=True,
                               transform=T.ToTensor())
    loader_val = DataLoader(mnist_val, batch_size=batch_size,
                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

    embed()
    # model = ResNet(10)
    # input  = t.autograd.Variable(t.randn(1, 3, 224, 224))
    # o = model(input)

    # print(o)
