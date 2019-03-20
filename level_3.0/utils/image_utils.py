#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/17
author: lujie
"""

import numpy as np
import os, tempfile
import matplotlib.pyplot as plt
from urllib import request, error
from scipy.misc import imread
from utils.fast_layers import conv_forward_fast



def visual_mini_dataset(data = None, classes_to_show = 7, examples_per_class = 5):
    ''' visual randomly selected iamge from target classes '''

    class_idxs = np.random.choice(len(data['class_names']), size=classes_to_show, replace=False)

    for i, class_idx in enumerate(class_idxs):

        train_idxs, = np.nonzero(data['y_train'] == class_idx)
        train_idxs  = np.random.choice(train_idxs, size=examples_per_class, replace=False)

        for j, train_idx in enumerate(train_idxs):

            img = deprocess_image(data['X_train'][train_idx], data['mean_image'])
            plt.subplot(examples_per_class, classes_to_show, 1 + i + classes_to_show * j)
            if j == 0:
                plt.title(data['class_names'][class_idx][0])
            plt.imshow(img)
            plt.gca().axis('off')
    plt.savefig('./mini_dataset.png', dpi=400); plt.close()




def blur_image(X):
    """
    A very gentle image blurring operation, to be used as a regularizer for image
    generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)
    """
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride': 1, 'pad': 1}
    for i in xrange(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


def preprocess_image(img, mean_img, mean='image'):
    """
    Convert to float, transepose, and subtract mean pixel

    Input:
    - img: (H, W, 3)

    Returns:
    - (1, 3, H, 3)
    """
    if mean == 'image': mean = mean_img

    elif mean == 'pixel': mean = mean_img.mean(axis=(1, 2), keepdims=True)

    elif mean == 'none': mean = 0

    else: raise ValueError('mean must be image or pixel or none')

    return img.astype(np.float32).transpose(2, 0, 1)[None] - mean


def deprocess_image(img, mean_img, mean = 'image', renorm = False):
    """
    Add mean pixel, transpose, and convert to uint8

    Input:
    - (1, 3, H, W) or (3, H, W)

    Returns:
    - (H, W, 3)
    """

    if mean == 'image': mean = mean_img
    elif mean == 'pixel': mean = mean_img.mean(axis=(1, 2), keepdims=True)
    elif mean == 'none': mean = 0
    else: raise ValueError('mean must be image or pixel or none')

    if img.ndim == 3:
        img = img[None]
    img = (img + mean)[0].transpose(1, 2, 0)

    if renorm:
        low, high = img.min(), img.max()
        img = 255.0 * (img - low) / (high - low)

    return img.astype(np.uint8)


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img

    except error.URLError as e:
        print('URL Error: ', e.reason, url)
    except error.HTTPError as e:
        print('HTTP Error: ', e.code, url)
