#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/17
author: lujie
"""

import numpy as np
import os, tempfile
import matplotlib.pyplot as plt
from IPython import embed
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


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, of shape (N, 3, H, W)
    - y: Labels for X, of shape (N,)
    - model: A PretrainedCNN that will be used to compute the saliency map.

    Returns:
    - saliency: An array of shape (N, H, W) giving the saliency maps for the input images.
    """
    saliency = None
    scores, cache = model.forward(X, mode='test')
    dscores = np.zeros(scores.shape)
    dscores[:, y] = 1

    w, _ = model.backward(dscores, cache)   # N x 3 x H x W
    saliency = np.max(np.abs(w),axis = 1)

    return saliency


def show_saliency_maps(data, model, num_show = 5):
    ''' show the saliency '''

    # mask = np.random.randint(data['X_val'].shape[0], size=num_show)
    mask = np.asarray([128, 3225, 2417, 1640, 4619])
    # mask = np.asarray(mask)
    X = data['X_val'][mask]
    y = data['y_val'][mask]

    saliency = compute_saliency_maps(X, y, model)

    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(X[i], data['mean_image']))
        plt.axis('off')
        plt.title(data['class_names'][y[i]][0])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i])
        plt.axis('off')
    plt.gcf().set_size_inches(10, 4)
    plt.show()


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies as target_y.

    Inputs:
    - X: Input image, of shape (1, 3, 64, 64)
    - target_y: An integer in the range [0, 100)
    - model: A PretrainedCNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y by the model.
    """

    X_fooling = X.copy()
    iter = 0; current_predict = -1

    while iter < 500 and current_predict != target_y:

        scores, cache = model.forward(X_fooling, mode='test')
        current_predict = scores[0].argmax()

        if current_predict != target_y:
            dscores = np.zeros(scores.shape)
            dscores[:,target_y] = max(scores[:,current_predict] - scores[:,target_y],100)
            dscores[:,current_predict] = min(-scores[:,current_predict] + scores[:,target_y],-100)
            w, _ = model.backward(dscores, cache)
            X_fooling += w
            iter += 1
        print("iteration: ", iter, "  predict: ", current_predict)

    return X_fooling


def show_fooling_iamge(data, model):
    ''' show the fooling image '''

    # Find a correctly classified validation image
    while True:
        i = np.random.randint(data['X_val'].shape[0])
        X = data['X_val'][i:i+1]
        y = data['y_val'][i:i+1]
        y_pred = model.loss(X)[0].argmax()
        if y_pred == y: break

    target_y = 67
    X_fooling = make_fooling_image(X, target_y, model)

    # Make sure that X_fooling is classified as y_target
    scores = model.loss(X_fooling)
    assert scores[0].argmax() == target_y, 'The network is not fooled!'

    embed()
    # Show original image, fooling image, and difference
    plt.subplot(1, 3, 1)
    plt.imshow(deprocess_image(X, data['mean_image']))
    plt.axis('off')
    plt.title(data['class_names'][y[0]][0])
    plt.subplot(1, 3, 2)
    plt.imshow(deprocess_image(X_fooling, data['mean_image'], renorm=True))
    plt.title(data['class_names'][target_y][0])
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Difference')
    plt.imshow(deprocess_image(X - X_fooling, data['mean_image']))
    plt.axis('off')
    plt.show()
