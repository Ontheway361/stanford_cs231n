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

from utils.data_utils import load_tiny_imagenet
from utils.gradient_check import *
from classifiers.pretrained_cnn import PretrainedCNN
from utils.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from utils.image_utils import image_from_url


def rel_error(x, y):
    """ returns relative error """

    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == '__main__':


    data = load_tiny_imagenet('../dataset/tiny-imagenet-100-A', subtract_mean=True)

    for i, names in enumerate(data['class_names']):
        print (i, ' '.join('"%s"' % name for name in names))

    classes_to_show = 7
    examples_per_class = 5

    class_idxs = np.random.choice(len(data['class_names']), size=classes_to_show, replace=False)

    embed()

    for i, class_idx in enumerate(class_idxs):

        train_idxs, = np.nonzero(data['y_train'] == class_idx)
        train_idxs = np.random.choice(train_idxs, size=examples_per_class, replace=False)
        for j, train_idx in enumerate(train_idxs):
            img = deprocess_image(data['X_train'][train_idx], data['mean_image'])
            plt.subplot(examples_per_class, classes_to_show, 1 + i + classes_to_show * j)
            if j == 0:
                plt.title(data['class_names'][class_idx][0])
            plt.imshow(img)
            plt.gca().axis('off')

    plt.show()
