#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/06
author: lujie
"""

import time
import json
import numpy as np
import configparser as cp
from IPython import embed
import matplotlib.pyplot as plt
from utils.solver import Solver
from classifiers.cnn import ConvNet
from utils.vis_utils import visualize_grid
from classifiers.fc_net import FullyConnectedNet
from config.architecture_config import Architecture


class ClassifierEngine(object):

    def __init__(self, method = 'fcn', configs = None):

        self.model = None

        if method is 'fcn':
            self.model = FullyConnectedNet(configs['arch'])

        elif method is 'cnn':
            self.model = ConvNet(configs['arch'])

        else: raise ValueError('Unrecognized classifier ...')

        self.solver = Solver(self.model, configs['solver'])

    def show_history(self):
        ''' show the history of training process '''

        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        plt.subplot(2, 1, 1)
        plt.plot(self.solver.loss_history)
        plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.title('Loss history')

        plt.subplot(2, 1, 2)
        plt.plot(self.solver.train_acc_history, label='train')
        plt.plot(self.solver.val_acc_history, label='val')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.title('Classification accuracy history')

        fig_version = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
        plt.savefig('../dataset/training_details/loss_curve/training_details_%s.png' % fig_version, dpi=400)
        # plt.show();
        plt.close()

    def visualize_filters(self):
        ''' visualize the filters '''

        grid = visualize_grid(self.model.params['W1'].transpose(0, 2, 3, 1))
        plt.imshow(grid.astype('uint8'))
        plt.axis('off')
        plt.gcf().set_size_inches(5, 5)
        fig_version = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
        plt.savefig('../dataset/training_details/filters/filters_%s.png' % fig_version, dpi=400)
        # plt.show();
        plt.close()

    def classifier_runner(self, verbose = True):
        '''
        step - 1. train a fcn_classifier
        step - 2. show the training-curve  [-optional-]
        step - 3. predict the test_date
        '''

        self.solver.train()

        self.solver.predict()

        if verbose:
            self.show_history()
            self.visualize_filters()


if __name__ == '__main__':

    method = 'cnn'
    arch = Architecture(method)
    configs = arch.get_configs(verbose = True)
    classifier_engine = ClassifierEngine(method, configs)
    start_time = time.time()
    classifier_engine.classifier_runner(verbose = False)
    finish_time = time.time()
    print('time cost : %.2f\n%s' % (finish_time - start_time, '-'*36))
