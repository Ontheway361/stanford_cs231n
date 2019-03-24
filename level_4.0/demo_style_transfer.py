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

from scipy.misc import imread
from collections import namedtuple
from utils.image_utils import *


if __name__ == '__main__':

    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor


    base_model = torchvision.models.squeezenet1_1(pretrained=True).features
    base_model.type(dtype)

    for param in base_model.parameters():
        param.requires_grad = False

    answers = dict(np.load('style-transfer-checks.npz'))

    content_loss_test(answers['cl_out'])
