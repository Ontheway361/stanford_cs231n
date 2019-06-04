#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/06/03
author: lujie
"""

import torch

class VGG(torch.nn.Module):

    def __init__(self, num_classes = 10, mode = 'D', init_weights = True):

        super(VGG, self).__init__()

        self.features = self._make_layers(mode, batch_norm=True)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = torch.nn.Sequential(

            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        ''' Initialize the weights '''

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


    @staticmethod
    def _make_layers(key = 'A', batch_norm = False):

        # [A:VGG-11, B:VGG-13, D:VGG-16, E:VGG-19]
        cfg_dict = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            }

        details = list()
        if key in cfg_dict:
            details = cfg_dict[key]
        else:
            raise TypeError('Unknow dict-key ...')

        layers, in_channels = [], 3
        for v in details:

            if v == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                in_channels = v

        features = torch.nn.Sequential(*layers)

        return features
