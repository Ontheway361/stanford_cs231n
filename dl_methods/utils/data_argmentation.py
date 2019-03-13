#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/13
author: lujie
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



class DataArgmentation(object):

    def __init__(self, X_data = None, y_data = None):
        '''
        raw_data : N x H x W x C
        '''
        # - Color Jitter
        self.X_data = X_data
        self.y_data = y_data

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

        filp_data = np.zeros_like(self.X_data)

        for index in range(self.X_data.shape[0]):
            filp_data[index] = cv2.flip(self.X_data[index], mode, dst=None)

        return flip_data


    def _trans(self, ratio = 0.10):
        '''
        Translation according to ratio, with choice :

        - 0 : trans up
        - 1 : trans down
        - 2 : trans left
        - 3 : trans right
        '''

        N, H, W, C = self.X_data.shape
        shift_H, shift_W = int(H * ratio), int(W * ratio)
        choice = np.random.randint(0, 4, N)
        trans_data = np.zeros_like(self.X_data)

        for direction in range(4):

            flag = choice == direction

            if direction == 0:
                trans_data[flag][:(H-shift_H), :, :] = self.X_data[flag][shift_H:, :, :]
            elif direction == 1:
                trans_data[flag][shift_H:, :, :] = self.X_data[flag][:(H-shift_H), :, :]
            elif direction == 2:
                trans_data[flag][:, :(W-shift_W), :] = self.X_data[flag][:, shift_W:, :]
            else:
                trans_data[flag][:, shift_W:, :] = self.X_data[flag][:, :(W-shift_W), :]

        return trans_data


    def _crop(self, ratio = 0.9, random_flag = False):
        '''
        Crop the image according to ratio

        - ratio : the ratio to crop from image
        - random_flag:
            - False : crop the center of image
            - True  : random crop
        '''

        N, H, W, C = self.X_data.shape
        H_crop, W_crop = int(H * ratio), int(W * ratio)
        H_range, W_range = H - H_crop, W - W_crop
        center_y, center_x = H_range // 2, W_range // 2
        crop_data = np.zeros_like(self.X_data)

        if not random_flag:
            crop_data[:, center_y:(center_y+H_crop), center_x:(center_x+W_crop), :] = \
                self.data[:, center_y:(center_y+H_crop), center_x:(center_x+W_crop), :]
        else:
            for index in range(N):
                y, x = np.random.randint(0, H_range), np.random.randint(0, W_range)
                crop_data[index, center_y:(center_y+H_crop), center_x:(center_x+W_crop), :] = \
                    self.X_data[index, y:(y+H_crop), x:(x+W_crop), :]

        return crop_data


    def _color_jitter(self, mode = 0):
        '''
        Color jitter according to mode

        - mode :
            - 0  : rgb -> hsv -> jitter -> rgb
            - 1  : rgb -> hsv
            - -1 : rgb -> gray
        '''

        color_data = np.zeros_like(self.X_data)

        if mode == 0:
        elif mode == 1:
        else:
