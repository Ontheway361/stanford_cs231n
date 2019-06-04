#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/06/04
author: lujie
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):

        super(Fire, self).__init__()

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)


    def forward(self, x):

        x      = self.squeeze_activation(self.squeeze(x))
        ea_1x1 = self.expand1x1_activation(self.expand1x1(x))
        ea_3x3 = self.expand3x3_activation(self.expand3x3(x))
        res = torch.cat([ea_1x1, ea_3x3], 1)

        return  res


class SqueezeNet(nn.Module):

    def __init__(self, num_classes = 10, mode = 'A'):

        super(SqueezeNet, self).__init__()

        if mode not in ['A', 'B']:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "A or B expected".format(version=version))

        self.num_classes = num_classes

        if mode == 'A':

            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )

        else:

            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        self.final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            self.final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self._initialize_weights()


    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), self.num_classes)

        return x
