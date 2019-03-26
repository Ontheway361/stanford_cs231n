#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/26
author: lujie
"""

import torch as t
from torch import  nn
from IPython import embed
from torch.nn import  functional as F


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
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def classifier(model, loader_train, num_epochs = 10, batch_size = 128):
    ''' train the resnet34 as a classifier '''

    iter_count = 0
    for epoch in range(num_epochs):

        for x, _ in loader_train:

            if len(x) != batch_size:
                continue
            score = model(x)
            loss = softmax_losss(score, y)
            loss.backward()
            model.step()
