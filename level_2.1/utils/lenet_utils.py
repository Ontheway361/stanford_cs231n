#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
date  : 2019/01/17
author: lujie
"""

import torch as t
from tqdm import tqdm
from torch.autograd import Variable


class LeNets(t.nn.Module):

    def __init__(self, num_class = 10):
        super(LeNets, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(3, 64, 7),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 5),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2)
        )
        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, 3),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2)
        )
        self.dense = t.nn.Sequential(
            t.nn.Linear(256, 128),
            t.nn.ReLU(),
            t.nn.Linear(128, num_class)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

    def loss_fun(self, net_out = None, label = None):
        '''
        '''
        loss_f = t.nn.CrossEntropyLoss()
        return loss_f(net_out, label)


def net_trainer(dataloader = None, num_epoch = 10):
    '''
    step - 0. check the status of GPU
    step - 1. set the optimizer and loss_func
    step - 2. training process go
    '''
  
    # step - 0
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


    # step - 1
    model = LeNets().to(device)
    optimizer = t.optim.Adam(model.parameters())

    # step - 2
    for epoch in range(num_epoch):

        train_loss = 0; train_acc = 0.0; num_train = 0

        for batch_x, batch_y in tqdm(dataloader):

            batch_x = Variable(batch_x, requires_grad=True).to(device)
            batch_y = Variable(batch_y).to(device)

            out = model(batch_x)
            loss = model.loss_fun(out, batch_y)
            train_loss += loss.item()
            pred = t.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            num_train += len(batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch %2d; \ttrain Loss: %.10f;\ttrain acc : %5.4f' % \
                   (epoch, train_loss/num_train, train_acc/num_train))

    return model


def net_infer(model = None, test_loader = None):
    ''' inference '''

    model = model.eval()
    eval_acc = 0.0; eval_loss = 0.0; num_test = 0

    for batch_x, batch_y in tqdm(test_loader):

        batch_x = Variable(batch_x, requires_grad=False).to(device)
        batch_y = Variable(batch_y).to(device)

        out = model(batch_x)
        loss = model.loss_fun(out, batch_y)
        eval_loss += loss.item()
        pred = t.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
        num_test += len(batch_y)

    print('test_loss : %.10f\ttest acc : %5.4f' % (eval_loss/num_test, eval_acc/num_test))
