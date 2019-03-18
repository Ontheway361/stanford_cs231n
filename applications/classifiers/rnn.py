#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/17
author: lujie
"""

import numpy as np

from utils.layers import *
from utils.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim = 512, wordvec_dim = 128, hidden_dim = 128, \
                     cell_type = 'rnn', dtype = np.float32):

        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """

        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null  = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end   = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim) / 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim) / np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim) / np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size) / np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        '''
        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params

        pipeline :
        step - 1. compute the initial hidden state with image features.
        step - 2. word embed according to W_embed
        step - 3. use rnn or lstm forward to generate the hidden_status
        step - 4. compute the temporal affine
        step - 5. calculate the loss
        '''

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self._null)

        # step - 1
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        initial_h = np.dot(features, W_proj) + b_proj

        # step - 2
        W_embed = self.params['W_embed']
        embed_word, embed_word_cache = word_embedding_forward(captions_in, W_embed)

        # step - 3
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        if self.cell_type=='rnn':
          h, h_cache = rnn_forward(embed_word, initial_h, Wx, Wh, b)
        elif self.cell_type =='lstm':
          h, h_cache = lstm_forward(embed_word, initial_h, Wx, Wh, b)

        # step - 4
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        affine_forward_out, affine_forward_cache = temporal_affine_forward(h, W_vocab, b_vocab)

        # step - 5
        loss, grads = 0.0, {}
        loss, dscore = temporal_softmax_loss(affine_forward_out, captions_out, mask, verbose=False)

        #backprop
        daffine_out, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dscore, affine_forward_cache)

        if self.cell_type=='rnn':
            dword_vector, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(daffine_out, h_cache)
        elif self.cell_type=='lstm':
            dword_vector, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(daffine_out, h_cache)

        grads['W_embed'] = word_embedding_backward(dword_vector, embed_word_cache)
        grads['W_proj'], grads['b_proj'] = features.T.dot(dh0), np.sum(dh0, axis=0)

        return loss, grads


    def reference(self, features, max_length = 30):
        '''
        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.

        pipeline :

        '''

        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        N, D = features.shape
        prev_h = features.dot(W_proj) + b_proj
        prev_c = np.zeros(prev_h.shape)

        # self._start is the index of the word '<START>'
        current_word_index = [self._start]*N

        for i in range(max_length):
            x = W_embed[current_word_index]  # get word_vector from word_index
            if self.cell_type=='rnn':
                next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
            elif self.cell_type =='lstm':
                next_h, next_c, _ = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
                prev_c = next_c
            prev_h = next_h
            next_h = np.expand_dims(next_h, axis=1)
            score, _ = temporal_affine_forward(next_h, W_vocab, b_vocab)
            captions[:,i] = list(np.argmax(score, axis = 2))
            current_word_index = captions[:,i]

        return captions
