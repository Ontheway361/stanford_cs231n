#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/24
author: lujie
"""

import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed
from scipy.misc import imread, imresize
from utils.gradient_check import *
from utils.data_utils import load_tiny_imagenet
from utils.image_utils import *
from classifiers.pretrained_cnn import PretrainedCNN


if __name__ == '__main__':

    data = load_tiny_imagenet('../../stanford_cs231n_dataset/tiny-imagenet-100-A', subtract_mean=True)

    model = PretrainedCNN(h5_file = '../../stanford_cs231n_dataset/pretrained_model.h5')


    #---------------------------------------------------------------------------#
    #             image generate according to label
    #---------------------------------------------------------------------------#
    # target_y = 43 # Tarantula
    # print(data['class_names'][target_y])
    # X = create_class_visualization(model, data, target_y, show_every=25)
    #---------------------------------------------------------------------------#

    #---------------------------------------------------------------------------#
    #             image generate according to feature_map
    #---------------------------------------------------------------------------#
    # filename = './kitten.jpg'
    # layer = 6
    # img = imresize(imread(filename), (64, 64))
    #
    # plt.ion()
    # plt.imshow(img)
    # plt.gcf().set_size_inches(3, 3)
    # plt.title('Original image')
    # plt.axis('off')
    # plt.pause(2)
    # plt.ioff()
    #
    # img_pre = preprocess_image(img, data['mean_image'])
    #
    # feats, _ = model.forward(img_pre, end=layer)
    #
    # kwargs = {
    #   'num_iterations': 2000,
    #   'learning_rate': 7500,
    #   'l2_reg': 1e-8,
    #   'show_every': 500,
    #   'blur_every': 200,
    # }
    # X = invert_features(feats, layer, model, data, **kwargs)
    #---------------------------------------------------------------------------#

    #---------------------------------------------------------------------------#
    #                                deep dream
    #---------------------------------------------------------------------------#
    filename = './kitten.jpg'
    max_size = 256

    img = image_scale(filename, max_size)

    plt.ion(); plt.imshow(img); plt.axis('off')
    plt.pause(2); plt.ioff()

    # Preprocess the image by converting to float, transposing,
    # and performing mean subtraction.
    img_pre = preprocess_image(img, data['mean_image'], mean='pixel')

    out = deepdream(img_pre, 7, model, data, learning_rate=2000)
