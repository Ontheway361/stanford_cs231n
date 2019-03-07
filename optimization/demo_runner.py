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
from utils.solver import Solver
from classifiers.fc_net import FullyConnectedNet

class FCN_Engine(object):

    def __init__(self, fcn_config = None, solver_config = None):

        self.model  = FullyConnectedNet(fcn_config)
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
        plt.show(); plt.close()

    def fcn_classifier(self, show_hist = True):
        '''
        step - 1. train a fcn_classifier
        step - 2. show the training-curve  [-optional-]
        step - 3. predict the test_date
        '''

        self.solver.train()

        if show_hist: self.show_history()

        self.solver.predict()
if __name__ == '__main__':

    #---------------------------------------------------------
    #                  parameters config
    #---------------------------------------------------------
    fcn_config = {
        'input_dim'     : 3 * 32 * 32,
        'hidden_dims'   : [100, 100, 100, 100],  # TODO
        'num_classes'   : 10,
        'dropout'       : 0.0,
        'use_batchnorm' : False,
        'weights_scale' : 2.5e-2,
        'reg'           : 2e-2,
        'dtype'         : np.float64,
        'seed'          : None
    }
    #---------------------------------------------------------
    solver_config = {
        'update_rule'   : 'adam',
        'learning_rate' : 3e-4,    # TODO
        'lr_decay'      : 0.90,
        'num_epochs'    : 10,      # TODO
        'batch_size'    : 64,      # TODO
        'verbose'       : True
    }
    #---------------------------------------------------------

    fcn_engine = FCN_Engine(fcn_config, solver_config)
    fcn_engine.fcn_classifier()
