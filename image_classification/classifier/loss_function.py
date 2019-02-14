#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/02/14
author: lujie
"""

import numpy as np

def softmax_loss_naive(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    dW_each = np.zeros_like(W)
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)    # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis = 1), (num_train, 1))   # 找到最大值然后减去，这样是为了防止后面的操作会出现数值上的一些偏差
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True) # N by C
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0
    for i in xrange(num_train):
        for j in xrange(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))    # 损失函数的公式L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]#梯度的公式 ∇Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk, where Pk = exp(fk)/∑j exp(fj
        dW += dW_each　　　　　　　　　　　　　　　　　　#这是把每个类的放在了一起
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)  # 加上正则
    dW /= num_traindW += reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # D by C
    num_train, dim = X.shape

    f = X.dot(W)    # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1), keepdims=True)
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0    # N by C
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)#向量化直接操作即可
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W

    return loss, dW

def svm_loss_naive(W, X, y, reg):
    """
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
         that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)   # initialize the gradient as zero
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:    #根据公式，正确的那个不用算
                continue
            margin = scores[j] - correct_class_score + 1   # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] += -X[i, :]     #  根据公式：∇Wyi Li = - xiT(∑j≠yi1(xiWj - xiWyi +1>0)) + 2λWyi
                dW[:, j] += X[i, :]         #  根据公式： ∇Wj Li = xiT 1(xiWj - xiWyi +1>0) + 2λWj , (j≠yi)
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.Inputs and outputs
    are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)   # initialize the gradient as zero
    scores = X.dot(W)        # N by C
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores_correct = scores[np.arange(num_train), y]   # 1 by N
    scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1
    margins = scores - scores_correct + 1.0     # N by C
    margins[np.arange(num_train), y] = 0.0
    margins[margins <= 0] = 0.0
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    # compute the gradient
    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1)                  # 1 by N
    margins[np.arange(num_train), y] = -row_sum
    dW += np.dot(X.T, margins)/num_train + reg * W     # D by C

    return loss, dW
