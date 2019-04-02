#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/26
author: lujie
"""

import torch as t
from torch import  nn
import numpy as np
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
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def loss(self, score, label):
        ''' loss function '''

        loss_function = t.nn.CrossEntropyLoss()
        loss_value = loss_function(score, label)

        return loss_value


def net_trainer(model, loader_train, num_epochs = 10, batch_size = 128):
    '''
    Train the resnet34 as a classifier

    step - 1. set the optimizer and loss_func
    step - 2. start the training process
    step - 3. return the trained_net as a classifier
    '''

    # step - 1
    optimizer = t.optim.Adam(model.parameters())

    # step - 2
    iter_count = 0
    for epoch in range(num_epochs):

        train_loss = 0; train_acc = 0.0; num_train = 0
        for batch_x, batch_y in loader_train:

            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            if len(batch_x) != batch_size:
                continue
            score = model(batch_x)
            loss  = model.loss(score, batch_y)
            num_train += len(batch_x)
            train_loss += loss.item()
            pred_y = t.max(score, 1)[1]
            logis = (pred_y == batch_y).sum()
            train_acc += logis.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch %2d; \ttrain Loss: %.10f;\ttrain acc : %5.4f' % \
                   (epoch, train_loss/num_train, train_acc/num_train))
    return model


def net_infer(model, loader_test):
    ''' test the net_classifier '''

    model = model.eval()
    eval_acc = 0.0; eval_loss = 0.0; num_test = 0
    for batch_x, batch_y in loader_test:

        batch_x, batch_y = Variable(batch_x, requires_grad=False), Variable(batch_y, requires_grad=False)
        score = model(batch_x)
        loss  = model.loss(score, batch_y)
        eval_loss += loss.item()
        pred = t.max(score, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
        num_test += len(batch_y)
    print('test_loss : %.10f\ttest acc : %5.4f' % (eval_loss/num_test, eval_acc/num_test))
