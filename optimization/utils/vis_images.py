#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/08
author: lujie
"""

import time
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from utils.solver import Solver
from classifiers.fc_net import FullyConnectedNet


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')



if __name__ == '__main__':
