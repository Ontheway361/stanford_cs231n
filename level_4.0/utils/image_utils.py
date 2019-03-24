#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/24
author: lujie
"""

from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
import torchvision.transforms as T
from scipy.misc import imread, imresize

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)



def preprocess(img, size=512):
    '''
    Normalize the input image,  unsqueeze demension

    Input:
    - img: (H, W, 3)

    Returns:
    - (1, 3, H, 3)
    '''
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(), std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    '''
    Denormalize, squeeze the dimension

    Input:
    - (1, 3, H, W) or (3, H, W)

    Returns:
    - (H, W, 3)
    '''

    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    ''' Rescale the image '''

    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def features_from_img(imgpath, imgsize):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(dtype)
    return extract_features(img_var, cnn), img_var


def blur_image(X):
    """
    A very gentle image blurring operation, to be used as a regularizer for
    image generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)
    """
    from cs231n.fast_layers import conv_forward_fast
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride': 1, 'pad': 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]],
                                  dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img
