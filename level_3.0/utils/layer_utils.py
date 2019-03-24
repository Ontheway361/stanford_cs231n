#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/02
author: lujie
"""

from utils.layers import *
from utils.fast_layers import *
from IPython import embed

def affine_relu_forward(x, w, b):
    '''
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    '''
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    ''' Backward pass for the affine-relu convenience layer '''

    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)   # TODO
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_params):
    ''' forward of affine -> bn -> relu '''

    affine_out, affine_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_params)
    relu_out, relu_cache = relu_forward(bn_out)
    cache = (affine_cache, bn_cache, relu_cache)

    return relu_out, cache


def affine_bn_relu_backward(dout, cache):
  """ backpropagetion of affine <- bn <- relu """

  affine_cache, bn_cache, relu_cache = cache
  dbn = relu_backward(dout, relu_cache)
  daffine, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
  dx, dw, db = affine_backward(daffine, affine_cache)

  return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, sandwich_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """

    a, conv_cache = conv_forward_fast(x, w, b, sandwich_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, sandwich_param)
    cache = (conv_cache, relu_cache, pool_cache)

    return out, cache


def conv_relu_pool_backward(dout, cache):
    ''' Backward pass for the conv-relu-pool convenience layer '''

    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)

    return dx, dw, db

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    ''' no pooling version '''

    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    ''' no pooling version '''
    
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

def conv_bn_relu_pool_forward(x, w, b, gamma, beta, bn_param, sandwich_param):
    '''
    Convenience layer that performs a convolution, batchnorm, a ReLU, and a pool.

    Input/Output like conv_relu_pool_forward
    '''

    a, conv_cache = conv_forward_fast(x, w, b, sandwich_param)
    bn_out, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(bn_out)
    out, pool_cache = max_pool_forward_fast(s, sandwich_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)

    return out, cache


def conv_bn_relu_pool_backward(dout, cache):
    ''' Backward pass for the sandwich_bn convenience layer '''

    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    da_bn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
    dx, dw, db = conv_backward_fast(da_bn, conv_cache)

    return dx, dw, db, dgamma, dbeta
