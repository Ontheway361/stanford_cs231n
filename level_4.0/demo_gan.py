#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/25
author: lujie
"""

import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.gan_helper import *


class ChunkSampler(sampler.Sampler):
    """ Samples elements sequentially from some offset.

    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':

    NUM_TRAIN = 50000
    NUM_VAL = 5000

    NOISE_DIM = 288  # 96 for mnist
    batch_size = 128
    #
    mnist_train = dset.MNIST('../../cs231n_dataset/MNIST_data', train=True, download=False,
                               transform=T.ToTensor())
    loader_train = DataLoader(mnist_train, batch_size=batch_size,
                              sampler=ChunkSampler(NUM_TRAIN, 0))

    mnist_val = dset.MNIST('../../cs231n_dataset/MNIST_data', train=True, download=False,
                               transform=T.ToTensor())
    loader_val = DataLoader(mnist_val, batch_size=batch_size,
                            sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

    dtype = torch.FloatTensor
    # dtype = t.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!


    params = {
    'adversarial': 'deep_conv',
    'loss_type'  : 'ls_gan',
    'show_every' : 250,
    'batch_size' : 128,
    'noise_size' : 96,
    'num_epochs' : 10,
    'dtype'      : dtype
    }
    gan_runner(loader_train, **params)
