#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/05/31
author: lujie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channals, **kwargs):

        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn   = nn.BatchNorm2d(out_channals)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):

        super(Inception, self).__init__()

        # 1x1 conv branch
        self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.b2_1x1_a = BasicConv2d(in_planes, n3x3red, kernel_size=1)
        self.b2_3x3_b = BasicConv2d(n3x3red, n3x3, kernel_size=3, padding=1)

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3_1x1_a = BasicConv2d(in_planes, n5x5red, kernel_size=1)
        self.b3_3x3_b = BasicConv2d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv2d(n5x5, n5x5, kernel_size=3, padding=1)    # what ?

        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_1x1  = BasicConv2d(in_planes, pool_planes, kernel_size=1)

    def forward(self, x):

        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        y  = torch.cat([y1, y2, y3, y4], 1)
        return y


class GoogLeNet(nn.Module):

    def __init__(self, num_class = 10):

        super(GoogLeNet, self).__init__()

        self.pre_layers = BasicConv2d(3, 192, kernel_size=3, padding=1)

        self.l3a = Inception(192,  64,  96, 128, 16, 32, 32)
        self.l3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.l4a = Inception(480, 192,  96, 208, 16,  48,  64)
        self.l4b = Inception(512, 160, 112, 224, 24,  64,  64)
        self.l4c = Inception(512, 128, 128, 256, 24,  64,  64)
        self.l4d = Inception(512, 112, 144, 288, 32,  64,  64)
        self.l4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.l5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.l5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):

        out = self.pre_layers(x)
        out = self.l3a(out)
        out = self.l3b(out)
        out = self.maxpool(out)
        out = self.l4a(out)
        out = self.l4b(out)
        out = self.l4c(out)
        out = self.l4d(out)
        out = self.l4e(out)
        out = self.maxpool(out)
        out = self.l5a(out)
        out = self.l5b(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def test():
    net = GoogLeNet()
    x = torch.randn(1,3,32,32)
    print(x)
    y = net(x)
    print(y)

test()
