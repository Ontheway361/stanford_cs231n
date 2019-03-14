#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/13
author: lujie
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from skimage import exposure, img_as_float
# from utils.data_utils import load_CIFAR10

class DataArgmentation(object):

    def __init__(self, dataset = None):
        '''
        dataset['X_train'] : N x H x W x C
        '''

        # self.data  = load_CIFAR10()
        self.data = dataset
        del dataset


    def _visual(self, img = None, normalize = True):
        ''' test the method '''

        if normalize:
            img_max, img_min = np.max(img), np.min(img)
            img = 255.0 * (img - img_min) / (img_max - img_min)
        plt.imshow(img.astype('uint8'))
        plt.gca().axis('off')


    def _flip(self, mode = 1):
        '''
        Flip the image according to mode :

        - 0  : vertical flip
        - 1  : horiontal flip
        - -1 : diag flip
        '''

        flip_data = np.zeros_like(self.data['X_train'])

        for index in range(self.data['X_train'].shape[0]):
            flip_data[index] = cv2.flip(self.data['X_train'][index], mode, dst=None)

        return flip_data


    def _trans(self, ratio = 0.10):
        '''
        Translation according to ratio, with choice :

        - 0 : trans up
        - 1 : trans down
        - 2 : trans left
        - 3 : trans right
        '''

        N, H, W, C = self.data['X_train'].shape
        shift_H, shift_W = int(H * ratio), int(W * ratio)
        choice = np.random.randint(0, 4, N)
        trans_data = np.zeros_like(self.data['X_train'])

        for direction in range(4):

            flag = choice == direction

            if direction == 0:
                trans_data[flag, :(H-shift_H), :, :] = self.data['X_train'][flag, shift_H:, :, :]
            elif direction == 1:
                trans_data[flag, shift_H:, :, :] = self.data['X_train'][flag, :(H-shift_H), :, :]
            elif direction == 2:
                trans_data[flag, :, :(W-shift_W), :] = self.data['X_train'][flag, :, shift_W:, :]
            else:
                trans_data[flag, :, shift_W:, :] = self.data['X_train'][flag, :, :(W-shift_W), :]

        return trans_data


    def _crop(self, ratio = 0.9, random_flag = True):
        '''
        Crop the image according to ratio

        - ratio : the ratio to crop from image
        - random_flag:
            - False : crop the center of image
            - True  : random crop
        '''

        N, H, W, C = self.data['X_train'].shape
        H_crop, W_crop = int(H * ratio), int(W * ratio)
        H_range, W_range = H - H_crop, W - W_crop
        center_y, center_x = H_range // 2, W_range // 2
        crop_data = np.zeros_like(self.data['X_train'])

        if not random_flag:
            crop_data[:, center_y:(center_y+H_crop), center_x:(center_x+W_crop), :] = \
                self.data['X_train'][:, center_y:(center_y+H_crop), center_x:(center_x+W_crop), :]
        else:
            for index in range(N):
                y, x = np.random.randint(0, H_range), np.random.randint(0, W_range)
                crop_data[index, center_y:(center_y+H_crop), center_x:(center_x+W_crop), :] = \
                    self.data['X_train'][index, y:(y+H_crop), x:(x+W_crop), :]

        return crop_data


    def _color_jitter(self, mode = 1, gamma = 0.5):
        '''
        Color jitter according to mode

        - mode :
            - -1 : rgb -> gray
            - 0  : rgb -> hsv
            - 1  : adjust brightness
            - 2  : rescale intensity
        '''

        N, H, W, C = self.data['X_train'].shape
        color_data = None

        if mode == -1:
            color_data = np.zeros(N, H, W)
            for index in range(N):
                color_data[index] = cv2.cvtColor(self.data['X_train'][index], cv2.COLOR_BGR2GRAY)   # TODO

        else:
            color_data = np.zeros_like(self.data['X_train'])

            if mode == 0:
                for index in range(N):
                    color_data[index] = cv2.cvtColor(self.data['X_train'][index], cv2.COLOR_BGR2HSV)

            elif mode == 1:
                for index in range(N):
                    color_data[index] = exposure.adjust_gamma(self.data['X_train'][index], gamma)    # gamma
                    # color_data[index] = exposure.adjust_log(self.X_data[index], gamma)   # log

            elif mode == 2:
                for index in range(N):
                    flag = exposure.is_low_contrast(self.data['X_train'][index])
                    if flag:
                        color_data[index] = exposure.rescale_intensity(self.data['X_train'][index])
                    else:
                        color_data[index] = self.data['X_train'][index]

            else:
                raise ValueError('Unrecognized color jitter mode')

        return color_data


    def _add_noise(self, scale = 0.01):
        ''' add the gaussian noise '''

        N, H, W, C = self.data['X_train'].shape
        noise_data = self.data['X_train'] + scale * np.random.randn(N, H, W, C)

        return noise_data


    def argumented(self, method_list = []):
        ''' generate the argumented data according to method_list '''

        y_data = self.data['y_train']

        res_data = []
        for method in method_list:

            if 'flip' in method:
                res_data.append(self._flip())

            elif 'color' in method:
                res_data.append(self._color_jitter())

            elif 'noise' in method:
                res_data.append(self._add_noise())

            elif 'crop' in method:
                res_data.append(self._crop())

            elif 'trans' in method:
                res_data.append(self._crop())
            else:
                print('Unrecognized argmented method : %s ...' % method)

        for index in range(len(res_data)):
            self.data['X_train'] = np.concatenate((self.data['X_train'], res_data[index]))
            self.data['y_train'] = np.concatenate((self.data['y_train'], y_data))

        order_list = np.arange(self.data['X_train'].shape[0])
        np.random.shuffle(order_list)
        self.data['X_train'] = self.data['X_train'][order_list]
        self.data['y_train'] = self.data['y_train'][order_list]

        # normalized
        sample_mean = np.mean(self.data['X_train'], axis = 0)
        sample_var  = np.var(self.data['X_train'], axis = 0)

        self.data['X_train'] = ((self.data['X_train'] - sample_mean) / np.sqrt(sample_var + 1e-4)).transpose(0, 3, 1, 2)
        self.data['X_val']   = ((self.data['X_val'] - sample_mean) / np.sqrt(sample_var + 1e-4)).transpose(0, 3, 1, 2)
        self.data['X_test']  = ((self.data['X_test'] - sample_mean) / np.sqrt(sample_var + 1e-4)).transpose(0, 3, 1, 2)

        return self.data


if __name__ == '__main__' :

    pass
