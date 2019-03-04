#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/02
author: lujie
"""

import time
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from utils.data_utils import load_CIFAR10
from utils.solver import Solver
from classifiers.fc_net import TwoLayerNet, FullyConnectedNet
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

class ClassifierEngine(object):
    def __init__(self):
        pass

pass

if __name__ == '__main__':

    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    print(x.shape, w.shape, b.shape)

    out, _ = affine_forward(x, w, b)
    correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                            [ 3.25553199,  3.5141327,   3.77273342]])
    
