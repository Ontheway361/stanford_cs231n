#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/05/31
author: lujie
"""

import sys
import time
import torch
import torchvision
from tqdm import tqdm
from config import config_setting
from utils.dataset import ToyData
from zoo import LeNets, AlexNet, VGG, GoogLeNet, ResNet, DenseNet, SqueezeNet

from IPython import embed


class DemoRunner(object):

    def __init__(self, args_dict = {}):

        self.args = config_setting()
        self._set_envs()


    def _set_envs(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python: {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch: {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("%s%s Configurations %s" % (str, self.args.base_net, str))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def _modelloader(self):
        ''' '''

        if self.args.base_net == 'lenet':
            self.base_model = LeNets(self.args.num_class)

        elif self.args.base_net == 'alexnet':
            self.base_model = AlexNet(self.args.num_class)

        elif self.args.base_net == 'vgg':
            self.base_model = VGG(self.args.num_class, self.args.mode)

        elif self.args.base_net == 'inception':
            self.base_model = GoogLeNet(self.args.num_class)

        elif self.args.base_net == 'resnet':
            self.base_model = ResNet(self.args.num_class)

        elif self.args.base_net == 'densenet':
            self.base_model = DenseNet(self.args.num_class, self.args.mode)

        elif self.args.base_net == 'squeezenet':
            self.base_model = SqueezeNet(self.args.num_class, self.args.mode)

        else:
            raise TypeError('Unknow base_net ...')
        print('=====> model %s loading finished =====>' % self.args.base_net)


    def _optimizer(self):
        ''' '''

        if self.args.platform == 'gpu':
            self.base_model = torch.nn.DataParallel(self.base_model, device_ids=self.args.gpus).cuda()
            torch.backends.cudnn.benchmark = True
        else:
            self.base_mode.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        if self.args.optim_md == 'sgd':
            self.optimizer = torch.optim.SGD(self.base_model.parameters(), self.args.lr, \
                                 momentum=self.args.momentum, weight_decay=5e-4, nesterov=True)
        elif self.args.optim_md == 'adam':
            self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), \
                                 eps=1e-8, weight_decay=5e-4)
        elif self.args.optim_md == 'rms':
            self.optimizer = torch.optim.RMSprop(self.base_model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            raise TypeError('Unknow optimizer, please check ...')

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 40], gamma=self.args.gamma)

    
    def _adjust_lr(self, epoch, exp_num):
        ''' Sets the learning rate to the initial LR decayed by 0.1 every 30 epochs '''

        # decay = self.args.gamma ** (sum(epoch >= np.array(self.args.lr_steps)))
        decay = self.args.gamma ** exp_num
        lr    = self.args.lr * decay
        decay = self.args.weight_decay

        for param_group in self.optimizer.param_groups:

            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']


    def _dataloader(self):
        ''' '''

        Data_driver = ToyData(self.args.__dict__)
        self.trainloader = Data_driver._dataloder(shuffle_flag=True)
        self.testloader  = Data_driver._dataloder(train_flag=False)

        print('=====> data loading finished =====>')


    def _train_engine(self, epoch):

        torch.set_grad_enabled(True)
        self.base_model.train()

        running_loss = 0; running_acc = 0.0
        start_time, num_instance = time.time(), 0
        for input, target in tqdm(self.trainloader):

            target     = target.cuda(async=True)
            target_var = target
            num_instance += len(target)
            output = self.base_model(input)
            loss   = self.criterion(output, target)

            running_loss += loss.item()
            pred_y = torch.max(output, 1)[1]
            logis = (pred_y == target).sum()
            running_acc += logis.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        end_time = time.time()
        print('epoch %2d; \ttrain Loss: %.10f;\ttrain acc : %5.4f' % \
                   (epoch, running_loss/num_instance, running_acc/num_instance))
        print('single-epoch time cost : %3d' % (end_time-start_time))


    def _valid_engine(self):

        torch.set_grad_enabled(False)
        self.base_model.eval()

        running_loss, running_acc, num_instance = 0, 0.0, 0
        for input, target in tqdm(self.testloader):

            target     = target.cuda(async=True)
            target_var = target
            num_instance += len(target)
            output = self.base_model(input)
            loss   = self.criterion(output, target)

            running_loss += loss.item()
            pred_y = torch.max(output, 1)[1]
            logis = (pred_y == target).sum()
            running_acc += logis.item()

        eval_loss, eval_acc = running_loss/num_instance, running_acc/num_instance
        print('test Loss: %.10f;\ttest acc : %5.4f' % (eval_loss, eval_acc))

        return eval_loss, eval_acc


    def _main_loop(self):

        print('=====> ready for training =====>')

        bst_loss, bst_acc = 1e3, -1
        saturate_cnt, exp_num = 0, 0

        for epoch in range(0, self.args.num_epochs):

            if saturate_cnt == self.args.num_saturate:
                exp_num += 1
                saturate_cnt = 0
                print('Attention!!!, lr decreases by a factor of %6d' % (10 ** exp_num))
            
            if self.args.adlr_style == 'parts':
                self._adjust_lr(epoch, exp_num)
            else:
                self.scheduler.step()

            self._train_engine(epoch)

            loss, acc = self._valid_engine()
            
            is_best = loss < bst_acc or acc > bst_acc

            if is_best:
                saturate_cnt = 0
                print('Yahoo, a new SOTA has been found ...')
            else:
                saturate_cnt += 1

            if self.args.save_flag and is_best:

                save_name = self.args.save_to + self.args.base_net + 'bst.pth.tar'
                torch.save({
                    'epoch'      : epoch+1,
                    'loss'       : bst_loss,
                    'acc'        : bst_acc,
                    'state_dict' : self.base_model.state_dict(),
                }, save_name)


    def _run(self):

        self._modelloader()

        self._optimizer()

        self._dataloader()

        self._main_loop()
