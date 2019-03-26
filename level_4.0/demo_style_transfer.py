#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/24
author: lujie
"""

import PIL
import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import torchvision.transforms as T

from IPython import embed

from utils.transfer_helper import *


if __name__ == '__main__':

    dtype = t.FloatTensor
    # dtype = torch.cuda.FloatTensor


    base_model = tv.models.squeezenet1_1(pretrained=True).features
    base_model.type(dtype)

    for param in base_model.parameters():
        param.requires_grad = False

    # answers = dict(np.load('./utils/style-transfer-checks.npz'))
    #
    # content_loss_test(answers['cl_out'], base_model)
    #
    # gram_matrix_test(answers['gm_out'], base_model)
    #
    # style_loss_test(answers['sl_out'], base_model)
    #
    # tv_loss_test(answers['tv_out'])

    params = {
    'content_image' : './styles/ret_campus.jpg',
    'style_image' : './styles/starry_night.jpg',
    'image_size' : 192,
    'style_size' : 192,
    'content_layer' : 3,
    'content_weight' : 5e-3, # 5e-2
    'style_layers' : (1, 4, 6, 7),   # (1, 4, 6, 7),
    'style_weights' : (20000, 500, 12, 1),
    'tv_weight' : 5e-2
    }

    style_transfer(model = base_model, **params)
