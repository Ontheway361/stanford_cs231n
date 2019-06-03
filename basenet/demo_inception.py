#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/05/31
author: lujie
"""

import time
import torch
import torchvision as tv
import torch.utils.data.dataloader as Data
from utils.inception_utils import GoogLeNet

from IPython import embed

class DemoRunner(object):

    def __init__(self, base_net = 'Inception', num_class = 10, num_epochs = 50):

        self.base_net     = base_net
        self.num_class    = num_class
        self.num_epochs   = num_epochs
        self.base_model   = None
        self.criterion    = None
        self.optimizer    = None
        self.trainloader  = None
        self.testloader   = None

    def _modelloader(self):
        ''' '''

        if self.base_net == 'Inception':
            self.base_model = GoogLeNet(self.num_class)
        else:
            raise TypeError('Unknow base_net ...')

        self.base_model = torch.nn.DataParallel(self.base_model, device_ids=[0, 1]).cuda()
        torch.backends.cudnn.benchmark = True

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), 1e-3, \
                             momentum=0.9, weight_decay=5e-4, nesterov=True)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
                             milestones=[30, 45], gamma=0.5)


    def _dataloader(self):
        ''' '''

        self.trainloader = Data.DataLoader(tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = True, \
                               transform = tv.transforms.Compose([
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ]), download = False),
                               batch_size=128, shuffle=True, num_workers=4, drop_last=True)

        self.testloader  = Data.DataLoader(tv.datasets.CIFAR10('../../cs231n_dataset/CIFAR10_data', train = False, \
                               transform = tv.transforms.Compose([
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ]), download = False),
                               batch_size=128, num_workers=4)


    def _train_engine(self, epoch):

        torch.set_grad_enabled(True)
        self.base_model.train()

        running_loss = 0; running_acc = 0.0
        num_instance = len(self.trainloader)
        for i, (input, target) in enumerate(self.trainloader):

            target     = target.cuda(async=True)
            target_var = target

            output = self.base_model(input)
            loss   = self.criterion(output, target_var)

            running_loss += loss.item()
            pred_y = torch.max(score, 1)[1]
            logis = (pred_y == batch_y).sum()
            running_acc += logis.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch %2d; \ttrain Loss: %.10f;\ttrain acc : %5.4f' % \
                   (epoch, running_loss/num_instance, running_acc/num_instance))

    def _valid_engine(self):

        torch.set_grad_enabled(False)
        self.base_model.eval()

        running_loss = 0; running_acc = 0.0
        num_instance = len(self.testloader)
        for i, (input, target) in enumerate(self.testloader):

            target     = target.cuda(async=True)
            target_var = target

            output = self.base_model(input)
            loss   = self.criterion(output, target_var)

            running_loss += loss.item()
            pred_y = torch.max(score, 1)[1]
            logis = (pred_y == batch_y).sum()
            running_acc += logis.item()

        print('test Loss: %.10f;\ttest acc : %5.4f' % \
                   (epoch, running_loss/num_instance, running_acc/num_instance))


    def _main_loop(self):

        for epoch in range(0, self.num_epochs):

            self.scheduler.step()

            self._train_engine(epoch)

            self._valid_engine()


    def _runner(self):

        self._modelloader()

        self._dataloader()

        self._main_loop()






if __name__ == '__main__':

    Inception = DemoRunner()

    Inception._runner()

    
