#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/10
author: lujie
"""

import numpy as np
from utils.layers import *
from utils.fast_layers import *
from utils.layer_utils import *
from IPython import embed

class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input channels.
    """

    def __init__(self, cnn_config = None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - fcn_layers: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        print('set parameters for convnets ... ')

        if not cnn_config: print('cnn_config is None, nets adopts default parameters ...')

        # set the default parameters for sandwich

        conv_layers = {
            'sandwich1' : {
                'num_filters' : 32,
                'filter_size' : 7,
                'padding'     : 'same',
                'stride'      : 1,
                'pool_height' : 2,
                'pool_width'  : 2,
                'pool_stride' : 2
            }
        }

        self.input_dim     = cnn_config.pop('input_dim', 3 * 32 * 32)
        self.conv_layers   = cnn_config.pop('conv_layers', conv_layers)
        self.fcn_layers    = cnn_config.pop('fcn_layers', [100])
        self.num_classes   = cnn_config.pop('num_classes', 10)
        self.use_batchnorm = cnn_config.pop('use_batchnorm', False)
        self.weight_scale  = cnn_config.pop('weight_scale', 2.5e-3)
        self.reg           = cnn_config.pop('reg', 1e-2)
        self.dtype         = cnn_config.pop('dtype', np.float32)
        self.num_layers    = len(self.conv_layers) + len(self.fcn_layers) + 1
        self.params        = {}

        if len(cnn_config) > 0:
            extra = ', '.join('"%s"' % k for k in cnn_config.keys())
            raise ValueError('Unrecognized arguments in cnn_config :  %s' % extra)

        self._init_params()


    def _init_params(self):
        ''' init the parameters for conv-layers '''

        for index in range(len(conv_layers)):

            sandwich_i = 'sandwich' + str(index + 1)
            if self.conv_layers[sandwich_i]['padding'] is 'same':
                self.conv_layers[sandwich_i]['padding'] = self.conv_layers[sandwich_i]['filter_size'] // 2

        # init cnn_params

        input_dim = self.input_dim

        for index in range(len(conv_layers)):

            C, H, W = input_dim
            filter_i, bias_i = 'W' + str(index + 1), 'b' + str(index + 1)
            sandwich_i = 'sandwich' + str(index + 1)

            self.params[filter_i] = self.weight_scale * np.random.randn(self.conv_layers[sandwich_i]['num_filters'], \
                                           C, self.conv_layers[sandwich_i]['filter_size'], self.conv_layers[sandwich_i]['filter_size'])
            self.params[bias_i] = np.zeros((self.conv_layers[sandwich_i]['num_filters'],))

            if self.use_batchnorm:

                gamma_i, beta_i = 'gamma' + str(index + 1), 'beta' + str(index + 1)
                self.params[gamma_i] = np.ones(self.conv_layers[sandwich_i]['num_filters'])
                self.params[beta_i] = np.zeros(self.conv_layers[sandwich_i]['num_filters'])

            const = 2 * self.conv_layers[sandwich_i]['padding'] - self.conv_layers[sandwich_i]['filter_size']

            H_conv_o = (H + const) // self.conv_layers[sandwich_i]['stride'] + 1
            W_conv_o = (W + const) // self.conv_layers[sandwich_i]['stride'] + 1
            H_pool_o = 1 + (H_conv_o - self.conv_layers[sandwich_i]['pool_height']) // self.conv_layers[sandwich_i]['pool_stride']
            W_pool_o = 1 + (W_conv_o -self.conv_layers[sandwich_i]['pool_width']) // self.conv_layers[sandwich_i]['pool_stride']

            input_dim = (self.conv_layers[sandwich_i]['num_filters'], H_conv_o, W_conv_o)


        # init fcn_params

        num_input = np.prod(input_dim)

        for index in range(len(conv_layers), self.num_layers - 1):

            weight_i, bias_i = 'W' + str(index + 1), 'b' + str(index + 1)
            self.params[weight_i] = self.weight_scale * np.random.randn(num_input, self.fcn_layers[index-len(conv_layers)])
            self.params[bias_i]   = np.zeros((self.fcn_layers[index-len(conv_layers)],))

            if self.use_batchnorm:

                gamma_i, beta_i = 'gamma' + str(index + 1), 'beta' + str(index + 1)
                self.params[gamma_i] = np.ones(self.fcn_layers[index-len(conv_layers)])
                self.params[beta_i] = np.zeros(self.fcn_layers[index-len(conv_layers)])

            num_input = self.fcn_layers[index-len(conv_layers)]


        # int last layer
        weight_i, bias_i = 'W' + str(self.num_layers), 'b' + str(self.num_layers)
        self.params[weight_i] = self.weight_scale * np.random.randn(num_input, self.num_classes)
        self.params[bias_i] = np.zeros((self.num_classes,))

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode' : 'train'} for i in range(self.num_layers - 1)]


        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                self.params[k] = v.astype(self.dtype)


    def _forward(self, output_list = None, cache_list = None):
        ''' forward process of architecture '''

        # cnn
        for index in range(1, len(self.conv_layers) + 1):

            sandwich_i = 'sandwich' + str(index)
            weight_i, bias_i = 'W' + str(index), 'b' + str(index)
            sandwich_param = self.conv_layers[sandwich_i]

            if self.use_batchnorm:
                gamma_i, beta_i = 'gamma' + str(index), 'beta' + str(index)
                output_list[index], cache_list[index] = conv_bn_relu_pool_forward(output_list[index - 1], self.params[weight_i], self.params[bias_i], \
                                                            self.params[gamma_i], self.params[bias_i], self.bn_params[index-1], sandwich_param)
            else:
                output_list[index], cache_list[index] = conv_relu_pool_forward(output_list[index - 1], self.params[weight_i], self.params[bias_i], sandwich_param)

        # fcn
        for index in range(len(self.conv_layers) + 1, self.num_layers):

            weight_i, bias_i = 'W' + str(index), 'b' + str(index)

            if self.use_batchnorm:
                gamma_i, beta_i = 'gamma' + str(index), 'beta' + str(index)
                output_list[index], cache_list[index] = affine_bn_relu_forward(output_list[index - 1], self.params[weight_i], self.params[bias_i], \
                                                            self.params[gamma_i], self.params[beta_i], self.bn_params[index-1])
            else:
                output_list[index], cache_list[index] = affine_relu_forward(output_list[index - 1], self.params[weight_i], self.params[bias_i])

        # last layer
        weight_i, bias_i = 'W' + str(self.num_layers), 'b' + str(self.num_layers)
        output_list[self.num_layers], cache_list[self.num_layers] = affine_forward(output_list[-2], self.params[weight_i], self.params[bias_i])


    def _backward(self, cache_list = None, grads = None, dy = None):
        ''' backward process of architecture '''

        dx_list= [None] * (self.num_layers + 1)
        dx_list[-1] = dy

        # backward for last layer
        weight_i, bias_i = 'W' + str(self.num_layers), 'b' + str(self.num_layers)
        dx_list[-2], grads[weight_i], grads[bias_i] = affine_backward(dx_list[-1], cache_list[-1])

        # backward for fcn
        for index in range(self.num_layers-1, self.num_layers-len(self.fcn_layers)-1, -1):

            weight_i, bias_i = 'W' + str(index), 'b' + str(index)

            if self.use_batchnorm:
                gamma_i, beta_i = 'gamma' + str(index), 'beta' + str(index)
                dx_list[index-1], grads[weight_i], grads[bias_i], grads[gamma_i], grads[beta_i] = affine_bn_relu_backward(dx_list[index], cache_list[index])
            else:
                dx_list[index-1], grads[weight_i], grads[bias_i] = affine_relu_backward(dx_list[index], cache_list[index])

        # backward for cnn
        for index in range(len(self.conv_layers), 0, -1):

            weight_i, bias_i = 'W' + str(index), 'b' + str(index)

            if self.use_batchnorm:
                gamma_i, beta_i = 'gamma' + str(index), 'beta' + str(index)
                dx_list[index-1], grads[weight_i], grads[bias_i], grads[gamma_i], grads[beta_i] = conv_bn_relu_pool_backward(dx_list[index], cache_list[index])
            else:
                dx_list[index-1], grads[weight_i], grads[bias_i] = conv_relu_pool_backward(dx_list[index], cache_list[index])


        dW3, dW2, dW1 = (dW3 + self.reg * W3), (dW2 + self.reg * W2), (dW1 + self.reg * W1)

        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        grads['W3'], grads['b3'] = dW3, db3




    def loss(self, X, y = None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        mode = 'test' if y is None else 'train'

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = None

        output_list, cache_list = [None] * (self.num_layers + 1), [None] * (self.num_layers + 1)
        output_list[0] = X

        self._forward(output_list, cache_list)

        scores = output_list[-1]

        if y is None: return scores

        loss, grads = 0, {}

        loss, dy = softmax_loss(scores, y)

        for index in range(self.num_layers):
            weight = self.params['W' + str(index + 1)]
            loss += 0.5 * self.reg * (np.sum(weight * weight)

        self._backward(cache_list, grads, dy)

        return loss, grads


pass
