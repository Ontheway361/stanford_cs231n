#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/06
author: lujie
"""

import time
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from utils.vis_utils import visualize_grid
from utils.solver import Solver
from classifiers.fc_net import FullyConnectedNet
from classifiers.cnn import ConvNet

class ClassifierEngine(object):

    def __init__(self, classifier = 'fcn', classifier_config = None, solver_config = None):

        self.model = None
        if classifier is 'fcn':
            self.model = FullyConnectedNet(classifier_config)
        elif classifier is 'cnn':
            self.model = ConvNet(classifier_config)
        else: raise ValueError('Unrecognized classifier ...')

        self.solver = Solver(self.model, solver_config)

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
        plt.savefig('./history_details/history_details_%s.png' % fig_version, dpi=400)
        # plt.show();
        plt.close()

    def visualize_filters(self):
        ''' visualize the filters '''

        grid = visualize_grid(self.model.params['W1'].transpose(0, 2, 3, 1))
        plt.imshow(grid.astype('uint8'))
        plt.axis('off')
        plt.gcf().set_size_inches(5, 5)
        fig_version = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
        plt.savefig('./filters_%s.png' % fig_version, dpi=400)
        # plt.show();
        plt.close()

    def classifier_runner(self, show_hist = True):
        '''
        step - 1. train a fcn_classifier
        step - 2. show the training-curve  [-optional-]
        step - 3. predict the test_date
        '''

        self.solver.train()

        if show_hist: self.show_history()

        self.solver.predict()

        self.visualize_filters()

if __name__ == '__main__':

    fcn_config = {
        'input_dim'     : 3 * 32 * 32,
        'hidden_dims'   : [1024, 100, 10],  # TODO
        'num_classes'   : 10,
        'dropout'       : 0.1,
        'use_batchnorm' : True,
        'weights_scale' : 2.5e-2,   #   5e-2
        'reg'           : 1e-2,
        'dtype'         : np.float64,
        'seed'          : None
    }

    cnn_config = {
        'input_dim'     : (3, 32, 32),
        'num_layers'    : 3,

        'conv_layers'   : {

            'sandwich1'    : {
                'num_filters' : 32,
                'filter_size' : 7,
                'padding'     : 'same',
                'stride'      : 1,
                'pool_height' : 2,
                'pool_width'  : 2,
                'pool_stride' : 2
            },

            'sandwich2'    : {
                'num_filters' : 32,
                'filter_size' : 3,
                'padding'     : 'same',
                'stride'      : 1,
                'pool_height' : 2,
                'pool_width'  : 2,
                'pool_stride' : 2
            }
        }

        'hidden_dim'    : 500,
        'num_classes'   : 10,
        'use_batchnorm' : False,
        'weight_scale'  : 2.5e-3,  # 2.5e-3
        'reg'           : 5e-3,
        'dtype'         : np.float32
    }

    solver_config = {
        'num_train'     : 1000,
        'update_rule'   : 'adam',
        'learning_rate' : 5e-4,    # TODO 5e-4
        'lr_decay'      : 0.95,
        'num_epochs'    : 10,      # TODO
        'batch_size'    : 100,     # TODO
        'verbose'       : True
    }
    #---------------------------------------------------------

    classifier_engine = ClassifierEngine('cnn', cnn_config, solver_config)
    classifier_engine.classifier_runner()
