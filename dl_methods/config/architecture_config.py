#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/13
author: lujie
"""

import numpy as np
from IPython import embed

class Architecture(object):

    def __init__(self, method = 'cnn'):

        self.art_cfg = None
        self.method  = method
        self.solver  = self._solver_config()
        if method is 'cnn':
            self.art_cfg = self._cnn_config()
        else:
            self.art_cfg = self._fcn_config()

    def _fcn_config(self):
        ''' config of fcn-architecture '''

        fcn_config = {
              'input_dim'     : 3 * 32 * 32,
              'hidden_dims'   : [1024, 100, 10],  # TODO
              'num_classes'   : 10,
              'dropout'       : 0.1,
              'use_batchnorm' : True,
              'weights_scale' : 2.5e-2,   #   5e-2
              'reg'           : 1e-2,
              'dtype'         : np.float64,
              'seed'          : None
          }

        return fcn_config


    def _cnn_config(self):
        ''' config of cnn-architecture '''

        architecture = {

              'input_dim'     : (3, 32, 32),

              'conv_layers'   : {

                  'sandwich1' : {
                      'num_filters' : 32,
                      'filter_size' : 7,
                      'padding'     : 'same',
                      'stride'      : 1,
                      'pool_height' : 2,
                      'pool_width'  : 2,
                      'pool_stride' : 2
                  },

                  # 'sandwich2' : {
                  #     'num_filters' : 1,
                  #     'filter_size' : 1,
                  #     'padding'     : 0,
                  #     'stride'      : 1,
                  #     'pool_height' : 1,
                  #     'pool_width'  : 1,
                  #     'pool_stride' : 1
                  # },
                  #
                  # 'sandwich3'  : {
                  #     'num_filters' : 32,
                  #     'filter_size' : 3,
                  #     'padding'     : 'same',
                  #     'stride'      : 1,
                  #     'pool_height' : 1,
                  #     'pool_width'  : 1,
                  #     'pool_stride' : 1
                  # },
                  #
                  # 'sandwich4'  : {
                  #     'num_filters' : 32,
                  #     'filter_size' : 3,
                  #     'padding'     : 'same',
                  #     'stride'      : 1,
                  #     'pool_height' : 2,
                  #     'pool_width'  : 2,
                  #     'pool_stride' : 2
                  # },
                  #
                  # 'sandwich5'  : {
                  #     'num_filters' : 1,
                  #     'filter_size' : 1,
                  #     'padding'     : 0,
                  #     'stride'      : 1,
                  #     'pool_height' : 1,
                  #     'pool_width'  : 1,
                  #     'pool_stride' : 1
                  # },

              },

              'fcn_layers'    : [500, 100],
              'num_classes'   : 10,
              'use_batchnorm' : True,
              'weight_scale'  : 2.5e-3,  # 2.5e-3
              'reg'           : 5e-3,
              'dtype'         : np.float32
          }

        return architecture


    def _solver_config(self):
        ''' config of solver '''

        solver_config = {
              'num_train'     : None,
              'argmented'     : [],    # ['flip', 'color', 'noise', 'trans', 'crop']
              'update_rule'   : 'adam',
              'learning_rate' : 5e-4,    # TODO 5e-4
              'lr_decay'      : 0.95,
              'num_epochs'    : 15,      # TODO
              'batch_size'    : 64,     # TODO
              'verbose'       : True
          }

        return solver_config


    def get_configs(self, verbose = True):
        ''' get the info of config '''

        if verbose:
            if self.method is 'cnn':
                print('%s conv-arch %s' % ('-'*66, '-'*66))
                for key, items in self.art_cfg.items():
                    if key is 'conv_layers':
                        for conv, arch in items.items():
                            print(conv, arch)
                    else:
                        print(key, items)
            else:
                print('%s fcns-arch %s' % ('-'*66, '-'*66))
                print(self.art_cfg)
            print('%s solver_config %s' % ('-'*64, '-'*64))
            print(self.solver)
            print('%s' % ('-'*143))

        res = {}
        res['arch']   = self.art_cfg
        res['solver'] = self.solver

        return res

if __name__ == '__main__':

    arch = Architecture()
    configs = arch.get_configs(verbose = True)
