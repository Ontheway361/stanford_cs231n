#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2019/02/17
author: lujie
"""

from IPython import embed
from utils.path_util import DATA_PATH
from utils.data_utils import load_CIFAR10
from classifier.knn_classifier import KNearestNeighbor
from classifier.linear_classifier import LinearSVM, Softmax

def cal_accuracy(gt_label = None, pred_label = None):
    '''
    used for calculating the accuracy of classifier
    '''
    return sum(gt_label == pred_label)/len(gt_label)

if __name__ == '__main__':

    # step - 1. load the dataset
    root_path = '%s/cifar-10-batches-py/' % DATA_PATH
    Xtr, Ytr, Xte, Yte = load_CIFAR10(root_path, memory_fit = False)
    num_train, num_test = Xtr.shape[0], Xte.shape[0]
    print('num_train : %6d;\tnum_test : %6d' % (num_train, num_test))

    # step - 2. call the knn_classifier
    # knn_classifier = KNearestNeighbor()
    # knn_classifier.train(Xtr.reshape(Xtr.shape[0], -1), Ytr)
    # knn_pred = knn_classifier.predict(Xte.reshape(num_test, -1), k = 10, num_loops = 0)
    # knn_acc = cal_accuracy(Yte, knn_pred)

    # step - 3. call the linear_svm
    linear_svm = LinearSVM()
    linear_svm.train(Xtr, Ytr, learning_rate = 1e-3, reg = 1e-5, num_iters = 100, batch_size = 200)
    svm_pred = linear_svm.predict(Xte, Yte)
    svm_acc  = cal_accuracy(Yte, svm_pred)

    # step - 4. call the softmax
    softmax_classifier = LinearSVM()
    softmax_classifier.train(Xtr, Ytr, learning_rate = 1e-3, reg = 1e-5, num_iters = 100, batch_size = 200)
    softmax_pred = softmax_classifier.predict(Xte, Yte)
    softmax_acc  = cal_accuracy(Yte, softmax_pred)

    print('knn_acc %6.4f;\tsvm_acc %6.4f;\tsoftmax_acc %6.4f' % (knn_acc, svm_acc, softmax_acc))
