#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/07
author: lujie
"""

import time
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from utils.data_utils import load_CIFAR10
from utils.solver import Solver
from utils.layers import *
from utils.optim import sgd, sgd_momentum, rmsprop, adam
from classifiers.fc_net import TwoLayerNet, FullyConnectedNet
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array


def rel_error(x, y):
    """ returns relative error """

    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__ == '__main__':

    #-----------------------------------------------------------------------------------
    #                                 check batchnorm_forward
    #-----------------------------------------------------------------------------------
    # N, D1, D2, D3 = 200, 50, 60, 3
    # X = np.random.randn(N, D1)
    # W1 = np.random.randn(D1, D2)
    # W2 = np.random.randn(D2, D3)
    # a = np.maximum(0, X.dot(W1)).dot(W2)
    #
    # print ('Before batch normalization:')
    # print ('  means: ', a.mean(axis=0))
    # print ('  stds: ', a.std(axis=0))
    #
    # print ('After batch normalization (gamma=1, beta=0)')
    # a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
    # print ('  mean: ', a_norm.mean(axis=0))
    # print ('  std: ', a_norm.std(axis=0))
    #
    #
    # gamma = np.asarray([1.0, 2.0, 3.0])
    # beta = np.asarray([11.0, 12.0, 13.0])
    # a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    # print ('After batch normalization (nontrivial gamma, beta)')
    # print ('  means: ', a_norm.mean(axis=0))
    # print ('  stds: ', a_norm.std(axis=0))
    #
    # N, D1, D2, D3 = 200, 50, 60, 3
    # W1 = np.random.randn(D1, D2)
    # W2 = np.random.randn(D2, D3)
    #
    # bn_param = {'mode': 'train'}
    # gamma = np.ones(D3)
    # beta = np.zeros(D3)
    # for t in range(500):
    #     X = np.random.randn(N, D1)
    #     a = np.maximum(0, X.dot(W1)).dot(W2)
    #     batchnorm_forward(a, gamma, beta, bn_param)
    #
    # bn_param['mode'] = 'test'
    # X = np.random.randn(N, D1)
    # a = np.maximum(0, X.dot(W1)).dot(W2)
    #
    # print('Before batch normalization (test-time):')
    # print('  means: ', a.mean(axis=0))
    # print('  stds: ', a.std(axis=0))
    #
    # a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)
    #
    # print('After batch normalization (test-time):')
    # print('  means: ', a_norm.mean(axis=0))
    # print('  stds: ', a_norm.std(axis=0))


    #-----------------------------------------------------------------------------------
    #                               check batchnorm_backward
    #-----------------------------------------------------------------------------------

    # N, D = 4, 5
    # x = 5 * np.random.randn(N, D) + 12
    # gamma = np.random.randn(D)
    # beta = np.random.randn(D)
    # dout = np.random.randn(N, D)
    #
    # bn_param = {'mode': 'train'}
    # fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
    # fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
    # fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]
    #
    # dx_num = eval_numerical_gradient_array(fx, x, dout)
    # da_num = eval_numerical_gradient_array(fg, gamma, dout)
    # db_num = eval_numerical_gradient_array(fb, beta, dout)
    #
    # _, cache = batchnorm_forward(x, gamma, beta, bn_param)
    # dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    # print('dx error: ', rel_error(dx_num, dx))
    # print('dgamma error: ', rel_error(da_num, dgamma))
    # print('dbeta error: ', rel_error(db_num, dbeta))

    # N, D = 100, 500
    # x = 5 * np.random.randn(N, D) + 12
    # gamma = np.random.randn(D)
    # beta = np.random.randn(D)
    # dout = np.random.randn(N, D)
    #
    # bn_param = {'mode': 'train'}
    # out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    #
    # t1 = time.time()
    # dx1, dgamma1, dbeta1 = batchnorm_backward_alt(dout, cache)
    # t2 = time.time()
    # dx2, dgamma2, dbeta2 = batchnorm_backward(dout, cache)
    # t3 = time.time()
    #
    # print('dx difference: ', rel_error(dx1, dx2))
    # print('dgamma difference: ', rel_error(dgamma1, dgamma2))
    # print('dbeta difference: ', rel_error(dbeta1, dbeta2))
    # print('speedup: %.2fx' % ((t2 - t1) / (t3 - t2)))

    #-----------------------------------------------------------------------------------
    #                               fcn with bn
    #-----------------------------------------------------------------------------------

    # N, D, H1, H2, C = 2, 15, 20, 30, 10
    # X = np.random.randn(N, D)
    # y = np.random.randint(C, size=(N,))
    #
    # fcn_configs = {
    #     'input_dim'     : D,
    #     'hidden_dims'   : [H1, H2],  # TODO
    #     'num_classes'   : C,
    #     'dropout'       : 0.0,
    #     'use_batchnorm' : True,
    #     'weights_scale' : 5e-2,
    #     'reg'           : 2e-2,
    #     'dtype'         : np.float64,
    #     'seed'          : None
    # }
    #
    # for reg in [0, 3.14]:
    #
    #     fcn_config = fcn_configs.copy()
    #
    #     print ('Running check with reg = ', reg)
    #     fcn_config['reg'] = reg
    #
    #     model = FullyConnectedNet(fcn_config)
    #
    #     # print(fcn_config)
    #
    #     loss, grads = model.loss(X, y)
    #     print ('Initial loss: ', loss)
    #
    #     for name in sorted(grads):
    #         f = lambda _: model.loss(X, y)[0]
    #         grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    #         print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))

    # x = np.random.randn(500, 500) + 10
    #
    # for p in [0.3, 0.6, 0.75]:
    #   out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
    #   out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})
    #
    #   print ('Running tests with p = ', p)
    #   print ('Mean of input: ', x.mean())
    #   print ('Mean of train-time output: ', out.mean())
    #   print ('Mean of test-time output: ', out_test.mean())
    #   print ('Fraction of train-time output set to zero: ', (out == 0).mean())
    #   print ('Fraction of test-time output set to zero: ', (out_test == 0).mean())

    # x = np.random.randn(10, 10) + 10
    # dout = np.random.randn(*x.shape)
    #
    # dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}
    # out, cache = dropout_forward(x, dropout_param)
    # dx = dropout_backward(dout, cache)
    # dx_num = eval_numerical_gradient_array(lambda x: dropout_forward(x, dropout_param)[0], x, dout)
    #
    # print('dx relative error: ', rel_error(dx, dx_num))

    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    fcn_configs = {
        'input_dim'     : D,
        'hidden_dims'   : [H1, H2],  # TODO
        'num_classes'   : C,
        'dropout'       : 0.0,
        'use_batchnorm' : True,
        'weights_scale' : 5e-2,
        'reg'           : 2e-2,
        'dtype'         : np.float64,
        'seed'          : 123
    }

    for dropout in [0, 0.25, 0.5]:
        print('Running check with dropout = ', dropout)

        fcn_config = fcn_configs.copy()

        fcn_config['dropout'] = dropout

        model = FullyConnectedNet(fcn_config)

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
