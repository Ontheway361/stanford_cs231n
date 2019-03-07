#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/02
author: lujie
"""

import numpy as np
from IPython import embed
from utils.layers import *
from utils.layer_utils import *


class TwoLayerNet(object):
    '''
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    '''

    def __init__(self, input_dim = 3*32*32, hidden_dim = 100, num_classes = 10, \
                     weight_scale = 1e-3, reg = 0.0):
        '''
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        '''

        self.params = {}
        self.reg = reg

        W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        b1 = np.zeros((1, hidden_dim))

        W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b2 = np.zeros((1, num_classes))

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

    def loss(self, X, y = None):
        '''
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        '''
        scores = None

        X1 = np.reshape(X, (X.shape[0], -1))

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # affine_relu_forward
        # step - 1. affine_forward
        # step - 2. relu
        X2, cache_2 = affine_relu_forward(X1, W1, b1)
        scores, cache_3 = affine_relu_forward(X2, W2, b2)

        if y is None: return scores

        loss, grads = 0, {}

        loss, dy = softmax_loss(scores, y)

        dX2, dW2, db2 = affine_relu_backward(dy, cache_3)
        dW2 += self.reg * W2
        dX1, dW1, db1 = affine_relu_backward(dX2, cache_2)  # TODO
        dW1 += self.reg * W1

        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2*W2))

        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, fcn_config = None):
        '''
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout = 0 then the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random initialization of the weights.
        - dtype: float32 is faster but less accurate, so you should use float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the model.
        '''

        if not fcn_config: print('fcn_config is None, nets adopts default parameters ...')

        self.input_dim     = fcn_config.pop('input_dim', 3 * 32 * 32)
        self.hidden_dims   = fcn_config.pop('hidden_dims', [100])
        self.num_classes   = fcn_config.pop('num_classes', 10)
        self.dropout       = fcn_config.pop('dropout', 0.0)
        self.use_batchnorm = fcn_config.pop('use_batchnorm', False)
        self.weights_scale = fcn_config.pop('weights_scale', 1e-2)
        self.reg           = fcn_config.pop('reg', 1e-2)
        self.dtype         = fcn_config.pop('dtype', np.float32)
        self.seed          = fcn_config.pop('seed', None)
        self.num_layers    = 1 + len(self.hidden_dims)  # add the output layer
        self.use_dropout   = self.dropout > 0
        self.params        = {}

        if len(fcn_config) > 0:
            extra = ', '.join('"%s"' % k for k in fcn_config.keys())
            raise ValueError('Unrecognized arguments in fcn_config :  %s' % extra)

        layer_dim = [self.input_dim] + self.hidden_dims + [self.num_classes]

        for i in range(1, self.num_layers + 1):

            Wi = 'W' + str(i)
            bi = 'b' + str(i)
            gammai = 'gamma' + str(i)
            betai = 'beta' + str(i)

            self.params[Wi] = self.weights_scale * np.random.randn(layer_dim[i - 1], layer_dim[i])
            self.params[bi] = np.zeros((layer_dim[i]))

            if self.use_batchnorm and i != self.num_layers:
                self.params[gammai] = np.ones(layer_dim[i])
                self.params[betai] = np.zeros(layer_dim[i])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': self.dropout}
            if self.seed is not None:
                self.dropout_param['seed'] = self.seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
          self.params[k] = v.astype(self.dtype)


    def loss(self, X, y = None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:    # Attention, this opp will update the self.bn_params
                bn_param["mode"] = mode

        scores = None

        affine_relu_cache= {}
        affine_bn_relu_cache = {}
        dropout_cache = {}
        input_x = X.reshape(X.shape[0], -1)    # TODO-BUG

        # layer 1 to self.num_layers - 1
        for i in range(1, self.num_layers):
            Wi = 'W' + str(i)
            bi = 'b' + str(i)

            if self.use_batchnorm:
                gammai = 'gamma' + str(i)
                betai = 'beta' + str(i)

                input_x, affine_bn_relu_cache[i] = affine_bn_relu_forward(input_x, self.params[Wi], self.params[bi], self.params[gammai], \
                                                                          self.params[betai], self.bn_params[i-1])
            else:
                input_x, affine_relu_cache[i] = affine_relu_forward(input_x, self.params[Wi], self.params[bi])

            if self.use_dropout:
                input_x, dropout_cache[i] = dropout_forward(input_x, self.dropout_param)

        # last layer
        Wi, bi = 'W' + str(self.num_layers), 'b' + str(self.num_layers)
        affine_out, affine_cache = affine_forward(input_x, self.params[Wi], self.params[bi])
        scores = affine_out

        if mode == 'test': return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)

        # last layer
        dXi, dWi, dbi = affine_backward(dscores, affine_cache)

        grads['W' + str(self.num_layers)] = dWi + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = dbi

        # other layers
        for i in range(self.num_layers - 1, 0, -1):

            if self.use_dropout:
                dXi = dropout_backward(dXi, dropout_cache[i])

            if self.use_batchnorm:

                dXi, dWi, dbi, dgammai, dbetai = affine_bn_relu_backward(dXi, affine_bn_relu_cache[i])
                grads['W' + str(i)] = dWi + self.reg * self.params['W' + str(i)]
                grads['b' + str(i)] = dbi
                grads['gamma' + str(i)] = dgammai
                grads['beta' + str(i)] = dbetai

            else:

                dXi, dWi, dbi = affine_relu_backward(dXi, affine_relu_cache[i])
                grads['W' + str(i)] = dWi + self.reg * self.params['W' + str(i)]
                grads['b' + str(i)] = dbi

        # add loss with regularization
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] * self.params['W' + str(i)])

        return loss, grads
