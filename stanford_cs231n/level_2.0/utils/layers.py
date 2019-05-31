#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/05
author: lujie
"""

from IPython import embed
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    X = np.reshape(x, (x.shape[0], -1))
    (N, D) = X.shape
    out = np.dot(X, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    (x, w, b) = cache

    x = x.reshape(x.shape[0], -1)  # TODO

    dx, dw, db = None, None, None

    N = x.shape[0]

    dx = np.dot(dout, w.T)

    dw = np.dot(x.T, dout)
    db = np.dot(dout.T, np.ones((N, 1)))
    db = np.reshape(db, b.shape)   # keep the shape as b

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(0, x)

    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = np.array(dout, copy = True)
    dx[x <= 0] = 0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):

    """
    -------------------------------------------------------------------------------
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    -------------------------------------------------------------------------------
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    -------------------------------------------------------------------------------
    step - 1. normalize the batch-samples
    step - 2. scale and shift the x_normalized
    step - 3. put the variables into cache
    -------------------------------------------------------------------------------
    """

    mode = bn_param['mode']
    eps  = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':

        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x, axis = 0)

        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_normalized + beta

        cache = (x, sample_mean, sample_var, x_normalized, beta, gamma, eps)    # TODO

        # update running_mean and runing_var
        bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * sample_mean
        bn_param['running_var'] = momentum * running_var + (1 - momentum) * sample_var

    elif mode == 'test':

        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta

    else: raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache


def batchnorm_backward(dout, cache):
    """
    -------------------------------------------------------------------------------
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    -------------------------------------------------------------------------------

    -------------------------------------------------------------------------------
    """
    dx, dgamma, dbeta = None, None, None

    (x, sample_mean, sample_var, x_normalized, beta, gamma, eps) = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(x_normalized*dout, axis = 0)
    dx_normalized = gamma * dout

    # just focus on high-level
    dsample_var = np.sum(-1.0/2*dx_normalized*(x-sample_mean)/(sample_var+eps)**(3.0/2), axis=0)
    dsample_mean = np.sum(-1/np.sqrt(sample_var+eps)* dx_normalized, axis=0) + 1.0/N*dsample_var *np.sum(-2*(x-sample_mean), axis=0)
    dx = 1/np.sqrt(sample_var+eps)*dx_normalized + dsample_var*2.0/N*(x-sample_mean) + 1.0/N*dsample_mean

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """

    p, mode = dropout_param['p'], dropout_param['mode']

    if 'seed' in dropout_param: np.random.seed(dropout_param['seed'])

    mask, out = None, None

    if mode == 'train':

        mask = (np.random.rand(*x.shape)) < (1-p)  # TODO
        out = mask * x

    elif mode == 'test': out = x * (1-p)

    else: raise ValueError('Invalid forward dropout mode "%s"' % mode)

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy = False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train': dx = mask * dout

    elif mode == 'test': dx = dout * p   # TODO

    else: raise ValueError('Invalid backward dropout mode %s' % mode)

    return dx


def conv(X, w, b, conv_param):
    '''
    X: shape (C, H, W)
    W: shape (C, HH, WW)
    b: float
    '''

    C, H, W = X.shape
    C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    # padding
    npad = ((0, 0), (pad, pad), (pad, pad))
    X = np.pad(X, pad_width = npad, mode = 'constant', constant_values = 0)

    # conv
    H_o = int(np.floor(1 + (H + 2 * pad - HH) / stride))
    W_o = int(np.floor(1 + (W + 2 * pad - WW) / stride))

    Y = np.zeros((H_o, W_o))

    for i in range(H_o):
        for j in range(W_o):
            left_top_y, left_top_x = i * stride, j * stride
            conv_map = X[:, left_top_y:(left_top_y + HH), left_top_x:(left_top_x + HH)] * w
            Y[i, j]  = np.sum(conv_map) + b

    return Y


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
      - 'pad'   : The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
                 H' = 1 + (H + 2 * pad - HH) // stride
                 W' = 1 + (W + 2 * pad - WW) // stride
    - cache: (x, w, b, conv_param)
    """

    out = None

    # get params
    N, C, H, W   = x.shape
    F, C, HH, WW = w.shape

    # conv for evry image
    out = []
    for i in range(N):
        channel_list = []
        for j in range(F):
            y = conv(x[i], w[j], b[j], conv_param)
            channel_list.append(y)
        out.append(channel_list)

    out = np.array(out)

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_o, W_o = dout.shape

    npad  = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_pad = np.pad(x, pad_width = npad, mode = 'constant', constant_values = 0)

    db = np.zeros((F))

    temp = dout.transpose(1, 0, 2,3)
    for f in range(F):
        db[f] = np.sum(temp[f, :, :, :])

    dw, dx_pad = np.zeros(w.shape), np.zeros(x_pad.shape)

    for n in range(N):
        for f in range(F):
            for i in range(H_o):
                for j in range(W_o):
                    current_x_matrix = x_pad[n, :, (i * stride):(i * stride + HH), (j * stride):(j * stride + WW)]
                    dw[f] = dw[f] + dout[n, f, i, j] * current_x_matrix
                    dx_pad[n, :, (i * stride):(i * stride + HH), (j * stride):(j * stride + WW)] += w[f] * dout[n, f, i, j]

    dx = dx_pad[:, :, pad: H + pad, pad: W + pad]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'] , pool_param['stride']

    H_out, W_out = int(np.floor(1 + (H - pool_height) / stride)), int(np.floor(1 + (W - pool_width) / stride))
    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    out[n, c, h, w] = np.max(x[n, c, (h * stride):(h * stride + pool_height), (w * stride):(w * stride + pool_width)])
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'] , pool_param['stride']
    N, C, H_out, W_out = dout.shape

    dx = np.zeros(x.shape)
    xin_index  = [(i,j) for i in range(N) for j in range(C)]
    dout_index = [(i,j) for i in range(H_out) for j in range(W_out)]
    pool_index = [(i,j) for i in range(pool_height) for j in range(pool_width)]

    for n, c in xin_index:
        for h, w in dout_index:
            current_matrix = x[n, c, (h * stride):(h * stride + pool_height), (w * stride):(w * stride + pool_width)]
            current_max = np.max(current_matrix)
            for (i,j) in pool_index:
                if current_matrix[i,j] == current_max:
                    dx[n, c, h*stride+i, w*stride+j] += dout[n,c,h,w]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    N, C, H, W = x.shape
    temp_output, cache = batchnorm_forward(x.transpose(0,3,2,1).reshape((N*H*W, C)), gamma, beta, bn_param)
    out = temp_output.reshape(N,W,H,C).transpose(0,3,2,1)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N,C,H,W = dout.shape
    dx_temp, dgamma, dbeta = batchnorm_backward(dout.transpose(0,3,2,1).reshape((N*H*W,C)),cache)
    dx = dx_temp.reshape(N,W,H,C).transpose(0,3,2,1)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis = 1)

    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    probs = np.exp(x - np.max(x, axis = 1, keepdims = True))
    probs /= np.sum(probs, axis = 1, keepdims = True)          # normalize

    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1   # why ?
    dx /= N
    return loss, dx
