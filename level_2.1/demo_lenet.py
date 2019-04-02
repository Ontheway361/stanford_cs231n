#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
date  : 2019/01/17
author: lujie
"""
import torch as t
import torchvision as tv
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
from utils.lenet_utils import LeNets, net_trainer, net_infer

from IPython import embed


if __name__ == '__main__':
    '''
    step - 1. load the train and test data
    step - 2. train the net
    step - 3. test the net
    '''
    # step - 1
    train_data = tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = True, \
                                         transform = tv.transforms.ToTensor(), download=False)
    test_data  = tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = False, \
                                         transform = tv.transforms.ToTensor())

    print("train_data:", len(train_data.data))
    print("test_data:", len(test_data.data))

    train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=128)

    # step - 2
    model = net_trainer(train_loader, 1)

    # step - 3
    net_infer(model, test_loader)


    # save all model | compute graph and parameters
    # t.save(model, '../../cs231n_dataset/trained_model/lenet.pkl')
    # del model
    #
    # model = t.load('../../cs231n_dataset/trained_model/lenet.pkl')
    # TestNet(model, test_loader)

    # save the parameters
    # t.save(model.state_dict(), '../../cs231n_dataset/trained_model/lenet_params.pkl')
    # del model
    # lenet = LeNets()
    # lenet.load_state_dict(t.load('../../cs231n_dataset/trained_model/lenet_params.pkl'))
    # TestNet(lenet, test_loader)
