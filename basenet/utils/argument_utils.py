#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/04/03
author: ReLU
"""


import torch
from PIL import Image
from IPython import embed
import matplotlib.pyplot as plt
from torchvision import transforms


class ArgumentData(object):
    
    def __init__(self):
        pass

    
    @staticmethod
    def train_aug(img, size = None):
        ''' Argument for training set '''
  
        size = img.shape[:2] if size is None else size     
  
        argumentor = transforms.Compose([

            transforms.Resize(size),
            transforms.RandomCrop(size),
            # transform.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            # transfroms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
   
        img = argumentor(img)
        
        return img
        
    @staticmethod
    def test_aug(img, size = None):
        ''' Argument for val/testing set '''
   
        size = img.shape[:2] if size is None else size
        
        argumentor = transforms.Compose([

            transforms.Resize(size),
            transforms.RandomCrop(size),
            # transform.CenterCrop(size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        img = argumentor(img)

        return img



if __name__ == '__main__':
   
    img = Image.open('./dog.png')
    aug_opp = ArgumentData()
    aug_img = aug_opp.train_aug(img, size=128)
    embed()
    print(img.size, aug_img.shape)
