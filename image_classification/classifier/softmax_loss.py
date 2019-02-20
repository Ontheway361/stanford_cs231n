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
    dW, dW_each = np.zeros_like(W), np.zeros_like(W)   # D by C
    num_train, dim = X.shape[0], X.shape[1]
    num_class = W.shape[1]
    f = X.dot(W)    # N by C

    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis = 1), (num_train, 1))        # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis = 1, keepdims = True) # N by C
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0

    for i in range(num_train):
        for j in range(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))           #      L = -(1/N)∑i∑jI(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :] # ∇Wk(L) = -(1/N)∑i xiT (I(k==yi)-Pk) + 2λWk, where Pk = exp(fk)/∑jexp(fj)
        dW += dW_each
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # D by C
    num_train, dim = X.shape[0], X.shape[1]
    f = X.dot(W)    # N by C

    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis = 1), (num_train, 1))   # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis = 1, keepdims = True)

    # drop the dummy value for log
    prob = np.where(prob > 1e-10, prob, 1e-10)
    
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0    # N by C
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W

    return loss, dW
