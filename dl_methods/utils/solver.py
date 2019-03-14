#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/04
author: lujie
"""

import os
import numpy as np
from utils import optim
from IPython import embed
from utils.data_utils import load_CIFAR10
from utils.data_argmentation import DataArgmentation

class Solver(object):

    def __init__(self, model = None, solver_config = None):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data with the following:
          'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
          'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
          'y_train': Array of shape (N_train,) giving labels for training images
          'y_val': Array of shape (N_val,) giving labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py. Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learningrate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed during training.
        """

        print('set parameters for solver ...')

        if not model: raise TypeError('model is None, please design the architecure of networks ...')

        if not solver_config: print('solver_config is None, nets adopts default parameters ...')

        self.model         = model
        self.data          = load_CIFAR10()
        self.argmented     = solver_config.pop('argmented', [])
        self.num_train     = solver_config.pop('num_train', None)
        self.update_rule   = solver_config.pop('update_rule', 'adam')
        self.learning_rate = solver_config.pop('learning_rate', 1e-3)
        self.lr_decay      = solver_config.pop('lr_decay', 1.0)
        self.batch_size    = solver_config.pop('batch_size', 100)
        self.num_epochs    = solver_config.pop('num_epochs', 20)
        self.verbose       = solver_config.pop('verbose', True)

        if len(solver_config) > 0:
            extra = ', '.join('"%s"' % k for k in solver_config.keys())
            raise ValueError('Unrecognized arguments in solver_config : %s' % extra)

        self._data_argmentation()

        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)

        if self.num_train:
            mask = np.random.choice(self.data['X_train'].shape[0], self.num_train, replace = True)
            self.data['X_train'] = self.data['X_train'][mask]
            self.data['y_train'] = self.data['y_train'][mask]

        self.update_rule = getattr(optim, self.update_rule)  # function call here

        self._init_history_details()


    def _init_history_details(self):
        ''' Set up some book-keeping variables for optimization. Don't call this manually. '''

        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {'learning_rate': self.learning_rate}
            self.optim_configs[p] = d


    def _data_argmentation(self):
        '''
        Data argmentation for generating a general X_train
        '''

        print('data argmentation method list :', self.argmented)

        argment_engine = DataArgmentation(self.data)

        self.data = argment_engine.argumented(self.argmented)

        # print('data argmentation is finished, num_train : %6d' % self.data['X_train'].shape[0])

    def batch_trainer(self):
        '''  forward & back_propagetion on single batch_instances '''

        # Make a minibatch of training data
        num_train = self.data['X_train'].shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size, replace = True)
        X_batch = self.data['X_train'][batch_mask]
        y_batch = self.data['y_train'][batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p_name, p_value in self.model.params.items():
            if isinstance(p_value, np.ndarray):
                dw = grads[p_name]
                config = self.optim_configs[p_name]
                next_w, next_config = self.update_rule(p_value, dw, config)
                self.model.params[p_name] = next_w
                self.optim_configs[p_name] = next_config


    def check_accuracy(self, X, y, num_samples = None, batch_size = 1000):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples, replace = True)
            N, X, y = num_samples, X[mask], y[mask]

        # Compute predictions in batches
        num_batches = int(N / batch_size)
        if N % batch_size != 0: num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):
        ''' Run optimization to train the model. '''

        print('training the classifier, please wait ...')

        num_train = self.data['X_train'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):

            self.batch_trainer()

            # At the end of every epoch, increment the epoch counter and decay the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0

            if epoch_end:
                self.epoch += 1
                for key in self.optim_configs:
                    self.optim_configs[key]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last iteration, and at the end of each epoch.
            if t == 0 or epoch_end:
                train_acc = self.check_accuracy(self.data['X_train'], self.data['y_train'], num_samples = 1000)
                val_acc   = self.check_accuracy(self.data['X_val'], self.data['y_val'])
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

            if self.verbose and epoch_end:
                print('cur_epoch : (%3d,%3d),\tloss : %.6f,\ttrain_acc : %.4f,\tval_acc : %.4f .' % \
                       (self.epoch, self.num_epochs, self.loss_history[-1], train_acc, val_acc))

            # Keep track of the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {}
                for key, value in self.model.params.items():
                    self.best_params[key] = value.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params


    def predict(self, batch_size = 1000):
        ''' predict the X_test with best_model '''

        del self.data['X_train'], self.data['y_train'], self.data['X_val'], self.data['y_val']

        num_batches = np.ceil(self.data['X_test'].shape[0] / batch_size).astype(int)
        acc_list = [0] * num_batches

        for index in range(num_batches):

            pred = np.argmax(self.model.loss(self.data['X_test'][:batch_size]), axis=1)
            acc  = (pred == self.data['y_test'][:batch_size]).mean()
            acc_list[index] = acc

            self.data['X_test'] = self.data['X_test'][batch_size:]
            self.data['y_test'] = self.data['y_test'][batch_size:]

        print(acc_list)
        test_acc = np.mean(acc_list)
        print('%s\ntest_acc : %6.4f\n%s' % ('-'*36, test_acc, '-'*36))
