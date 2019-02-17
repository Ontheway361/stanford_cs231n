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


def cal_accuracy(gt_label = None, pred_label = None):
    '''
    used for calculating the accuracy of classifier
    '''
    return sum(gt_label == pred_label)/len(gt_label)

if __name__ == '__main__':

    # step - 1. load the dataset
    root_path = '%s/cifar-10-batches-py/' % DATA_PATH
    Xtr, Ytr, Xte, Yte = load_CIFAR10(root_path, memory_fit = True)
    num_train, num_test = Xtr.shape[0], Xte.shape[0]
    print('num_train : %6d;\tnum_test : %6d' % (num_train, num_test))

    # step - 2. call the knn_classifier
    knn_classifier = KNearestNeighbor()
    knn_classifier.train(Xtr.reshape(Xtr.shape[0], -1), Ytr)
    knn_pred = knn_classifier.predict(Xte.reshape(num_test, -1), k = 10, num_loops = 0)
    knn_acc = cal_accuracy(Yte, knn_pred)

    # step - 3. call the 
