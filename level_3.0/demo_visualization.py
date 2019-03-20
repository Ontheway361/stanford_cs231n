#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/19
author: lujie
"""

import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

from utils.gradient_check import *
from utils.data_utils import load_tiny_imagenet
from utils.image_utils import visual_mini_dataset
from classifiers.pretrained_cnn import PretrainedCNN


if __name__ == '__main__':

    data = load_tiny_imagenet('../dataset/tiny-imagenet-100-A', subtract_mean=True)

    # classes_to_show, examples_per_class = 7, 5
    #
    # visual_mini_dataset(data, classes_to_show, examples_per_class)

    model = PretrainedCNN(h5_file='../datasets/pretrained_model.h5')

    batch_size = 100
    # Test the model on training data
    mask = np.random.randint(data['X_train'].shape[0], size=batch_size)
    X, y = data['X_train'][mask], data['y_train'][mask]
    y_pred = model.loss(X).argmax(axis=1)
    print('Training accuracy: ', (y_pred == y).mean())

    # Test the model on validation data
    mask = np.random.randint(data['X_val'].shape[0], size=batch_size)
    X, y = data['X_val'][mask], data['y_val'][mask]
    y_pred = model.loss(X).argmax(axis=1)
    print('Validation accuracy: ', (y_pred == y).mean())
