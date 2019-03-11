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
      - hidden_dim: Number of units to use in the fully-connected hidden layer
      - num_classes: Number of scores to produce from the final affine layer.
      - weight_scale: Scalar giving standard deviation for random initialization of weights.
      - reg: Scalar giving L2 regularization strength
      - dtype: numpy datatype to use for computation.
      """

      print('set parameters for ConvNets ... ')

      if not cnn_config: print('cnn_config is None, nets adopts default parameters ...')

      # set the default parameters for sandwich

      default_sandwich = {
          'num_filters' : 32,
          'filter_size' : 7,
          'padding'     : 'same',
          'stride'      : 1,
          'pool_height' : 2,
          'pool_width'  : 2,
          'pool_stride' : 2,
      }

      self.input_dim     = cnn_config.pop('input_dim', 3 * 32 * 32)
      self.num_layers    = cnn_config.pop('num_layers', 3)
      self.sandwich1     = cnn_config.pop('sandwich1', default_sandwich)
      self.hidden_dim    = cnn_config.pop('hidden_dim', 100)
      self.num_classes   = cnn_config.pop('num_classes', 10)
      self.use_batchnorm = cnn_config.pop('use_batchnorm', False)
      self.weight_scale  = cnn_config.pop('weight_scale', 2.5e-3)
      self.reg           = cnn_config.pop('reg', 1e-2)
      self.dtype         = cnn_config.pop('dtype', np.float32)
      self.params        = {}

      if len(cnn_config) > 0:
          extra = ', '.join('"%s"' % k for k in cnn_config.keys())
          raise ValueError('Unrecognized arguments in cnn_config :  %s' % extra)

      # fisrt sandwich
      if self.sandwich1['padding'] is 'same':
          pad = (self.sandwich1['filter_size'] - 1) // 2
      else:
          pad = self.sandwich1['padding']
      conv1_param = {'stride': self.sandwich1['stride'], 'pad': pad}
      pool1_param = {'pool_height' : self.sandwich1['pool_height'], \
                    'pool_width'  : self.sandwich1['pool_width'], \
                    'pool_stride' : self.sandwich1['pool_stride']}

      C, H, W = self.input_dim

      # params for conv_pool layer
      self.params['W1'] = self.weight_scale * np.random.randn(self.sandwich1['num_filters'], \
                          C, self.sandwich1['filter_size'], self.sandwich1['filter_size'])
      self.params['b1'] = np.zeros((self.sandwich1['num_filters'],))
      self.params['conv1_param'] = conv1_param
      self.params['pool1_param'] = pool1_param

      H_conv_o = 1 + (H + 2 * conv1_param['pad'] - self.sandwich1['filter_size']) // conv1_param['stride']
      W_conv_o = 1 + (W + 2 * conv1_param['pad'] - self.sandwich1['filter_size']) // conv1_param['stride']
      H_pool_o = 1 + (H_conv_o - pool1_param['pool_height']) // pool1_param['pool_stride']
      W_pool_o = 1 + (W_conv_o - pool1_param['pool_width']) // pool1_param['pool_stride']

      # params for the second layer - affine
      num_input = self.sandwich1['num_filters'] * H_pool_o * W_pool_o


      # fully connected layer
      self.params['W2'] = self.weight_scale * np.random.randn(num_input, self.hidden_dim)
      self.params['b2'] = np.zeros((self.hidden_dim,))

      # params for the third layer - affine
      num_input = self.hidden_dim
      num_output = self.num_classes
      self.params['W3'] = self.weight_scale * np.random.randn(num_input, num_output)
      self.params['b3'] = np.zeros((num_output,))

      self.bn_params = []
      if self.use_batchnorm:

          self.bn_params = [{'mode' : 'train'} for i in range(self.num_layers-1)]

          self.params['gamma1'] = np.ones(self.sandwich1['num_filters'])
          self.params['beta1']  = np.zeros(self.sandwich1['num_filters'])

          self.params['gamma2'] = np.ones(self.hidden_dim)
          self.params['beta2']  = np.zeros(self.hidden_dim)

      for k, v in self.params.items():
          if isinstance(v, np.ndarray):
              self.params[k] = v.astype(self.dtype)


  def loss(self, X, y = None):
      """
      Evaluate loss and gradient for the three-layer convolutional network.
      Input / output: Same API as TwoLayerNet in fc_net.py.
      """

      mode = 'test' if y is None else 'train'

      if self.use_batchnorm:
          for bn_param in self.bn_params:
              bn_param["mode"] = mode

      W1, b1 = self.params['W1'], self.params['b1']
      W2, b2 = self.params['W2'], self.params['b2']
      W3, b3 = self.params['W3'], self.params['b3']

      # pass conv_param to the forward pass for the convolutional layer
      filter_size = W1.shape[2]
      conv1_param = self.params['conv1_param']

      # pass pool1_param to the forward pass for the max-pooling layer
      pool1_param = self.params['pool1_param']

      scores = None

      # forward
      Y_1, cache_1 = None,  None
      Y_2, cache_2 = None,  None
      Y_3, cache_3 = None,  None

      if self.use_batchnorm:
          Y_1, cache_1 = conv_bn_relu_pool_forward(X, W1, b1, self.params['gamma1'], \
                                             self.params['beta1'], self.bn_params[0], conv1_param, pool1_param)
          Y_2, cache_2 = affine_bn_relu_forward(Y_1, W2, b2, self.params['gamma2'], self.params['beta2'], self.bn_params[1])
      else:
          Y_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv1_param, pool1_param)  # TODO
          Y_2, cache_2 = affine_relu_forward(Y_1, W2, b2)

      Y_3, cache_3 = affine_forward(Y_2, W3, b3)

      scores = Y_3

      if y is None: return scores

      loss, grads = 0, {}

      loss, dy = softmax_loss(scores, y)

      loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

      # backward
      dx3, dW3, db3 = None, None, None
      dx2, dW2, db2 = None, None, None
      dx1, dW1, db1 = None, None, None

      dx3, dW3, db3 = affine_backward(dy, cache_3)

      if self.use_batchnorm:
          dx2, dW2, db2, dgamma2, dbeta2 = affine_bn_relu_backward(dx3, cache_2)
          dx1, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_pool_backward(dx2, cache_1)
          grads['gamma2'], grads['beta2'] =  dgamma2, dbeta2
          grads['gamma1'], grads['beta1'] =  dgamma1, dbeta1
      else:
          dx2, dW2, db2 = affine_relu_backward(dx3, cache_2)
          dx1, dW1, db1 = conv_relu_pool_backward(dx2, cache_1)   # TODO

      dW3, dW2, dW1 = (dW3 + self.reg * W3), (dW2 + self.reg * W2), (dW1 + self.reg * W1)

      grads['W1'], grads['b1'] = dW1, db1
      grads['W2'], grads['b2'] = dW2, db2
      grads['W3'], grads['b3'] = dW3, db3

      return loss, grads


pass
