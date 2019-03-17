#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/17
author: lujie
"""



import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from utils.rnn_layers import *
from utils.captioning_solver import CaptioningSolver
from classifiers.rnn import CaptioningRNN
from utils.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from utils.image_utils import image_from_url


def rel_error(x, y):
    """ returns relative error """

    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == '__main__':

# ------------------------------------
# train_captions
# train_image_idxs
# val_captions
# val_image_idxs
# train_features  #
# val_features
# idx_to_word
# word_to_idx     #
# train_urls
# val_urls
# ------------------------------------


    data = load_coco_data(max_train=50)

    small_rnn_model = CaptioningRNN(
              cell_type='rnn',
              word_to_idx = data['word_to_idx'],
              input_dim = data['train_features'].shape[1],
              hidden_dim=512,
              wordvec_dim=256,
            )

    small_rnn_solver = CaptioningSolver(small_rnn_model, data,
               update_rule='adam',
               num_epochs=50,
               batch_size=25,
               optim_config={
                 'learning_rate': 5e-3,
               },
               lr_decay=0.95,
               verbose=True, print_every=10,
             )

    small_rnn_solver.train()

    # Plot the training losses
    plt.plot(small_rnn_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()
