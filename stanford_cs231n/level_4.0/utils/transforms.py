#!/usr/bin/env python
# coding=utf-8

import os,time
import numpy as np
import cv2
import torch
import torchvision
import random
import scipy.io
import tensorflow as tf
from glob import glob
import scipy.ndimage as ndimage
from IPython import embed
from config import *


def FancyPCA(original_image):
    h, w, c = original_image.shape
    renorm_image = np.reshape(original_image, (h * w, c))
    renorm_image = renorm_image.astype('float32')
    mean = np.mean(renorm_image, axis=0)
    std = np.std(renorm_image, axis=0)
    renorm_image -= mean
    renorm_image /= std

    cov = np.cov(renorm_image, rowvar=False)

    lambdas, p = np.linalg.eig(cov)
    alphas = np.random.normal(0, 0.1, c)
    delta = np.dot(p, alphas*lambdas)

    pca_renorm_image = renorm_image + delta
    pca_color_image = pca_renorm_image * std + mean
    pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype('float32')
    return pca_color_image.reshape(h, w, c)

def SaltAndPepper(src,percetage=0.001):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])            
    for i in range(NoiseNum):
        randX=np.random.randint(0,src.shape[0]-1)
        randY=np.random.randint(0,src.shape[1]-1)
        if np.random.randint(0,1)==0:
            NoiseImg[randX,randY,:]=0
        else:
            NoiseImg[randX,randY,:]=255
    return NoiseImg

class PadResize(object):
    def __init__(self, size=(256, 256)):
        self.size = size
    
    def __call__(self, image, depth, mask=None):
        h, w, c = image.shape
        if h == w:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, self.size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
            return image, depth, mask
        if h > w:
            maxw = h
            t = 0
            l = int((h-w)/2)
        else:
            maxw = w
            t = int((w-h)/2)
            l = 0
        expand_image = np.zeros((maxw, maxw, c), dtype=image.dtype)
        expand_image[t:t+h, l:l+w] = image
        expand_depth = np.zeros((maxw, maxw), dtype=depth.dtype)
        expand_depth[t:t+h, l:l+w] = depth
        expand_mask = np.zeros((maxw, maxw), dtype=mask.dtype)
        expand_mask[t:t+h, l:l+w] = mask
        image = cv2.resize(expand_image, self.size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(expand_depth, self.size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(expand_mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, depth, mask

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, landmarks, mask):
        if np.random.randn() > 0.5:
            image = image[:,::-1,:]
            landmarks = landmarks[:,::-1]
            mask = mask[:,::-1]

        return image, landmarks, mask


class RandomCropRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    A crop is done to keep same image ratio, and no black pixels
    angle: max angle of the rotation, cannot be more than 180 degrees
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, angle=10, diff_angle=0, order=2, mode='constant'):
        self.angle = angle
        self.order = order
        self.diff_angle = diff_angle
        self.mode = mode

    def __call__(self, inputs, target, mask):
        h, w, _ = inputs.shape

        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2

        angle1_rad = angle1 * np.pi / 180
        inputs = ndimage.interpolation.rotate(inputs, angle1, reshape=True, order=self.order, mode=self.mode)
        target = ndimage.interpolation.rotate(target, angle1, reshape=True, order=0, mode=self.mode)
        mask   = ndimage.interpolation.rotate(mask, angle1, reshape=True, order=0, mode=self.mode)
        # keep angle1 and angle2 within [0,pi/2] with a reflection at pi/2: -1rad is 1rad, 2rad is pi - 2 rad
        angle1_rad = np.pi / 2 - np.abs(angle1_rad % np.pi - np.pi / 2) # still angle1_rad

        c1 = np.cos(angle1_rad)
        s1 = np.sin(angle1_rad)
        c_diag = h / np.sqrt(h * h + w * w)
        s_diag = w / np.sqrt(h * h + w * w)

        ratio = 1. / (c1 + w / float(h) * s1) # cause H = w*s1+h*c1, so h/H = ratio

        crop = CropCenter((int(h * ratio), int(w * ratio)))
        return crop(inputs, target, mask)


class CropCenter(object):   # object is center is better
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, target, mask):
        h1, w1, _ = inputs.shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        assert x1 >= 0 and y1 >= 0, "Crop size must larger than size"
        inputs = inputs[y1 : y1 + th, x1 : x1 + tw]
        target = target[y1 : y1 + th, x1 : x1 + tw]
        mask   = mask[y1 : y1 + th, x1 : x1 + tw]
        return inputs, target, mask


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=256):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, real, mask):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        real = real[top: top + new_h, left: left + new_w]
        mask = mask[top: top + new_h, left: left + new_w]
        return image, real, mask


class Bgr2Yuv(object):
    def __call__(self, image, labels, mask):
        # cv2.imwrite('rgb.jpg', image)
        yuv = np.clip(image.copy(), 0, 255).astype(np.float32)
        yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
        yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
        yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
        image = np.clip(yuv, 0, 255)
        # cv2.imwrite('yuv.jpg', image[:,:,(2,1,0)])
        '''
        # color trans image.dtype must be uint8
        if image.dtype == 'uint8':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            cv2.imwrite('yuv.jpg', image[:,:,(2,1,0)])
        else:
            print('error data type in Bgr2Yuv')
        '''
        return image.astype('uint8'), labels, mask

class Resize(object):
    def __init__(self, size=(256,256)):
        self.size = size
    def __call__(self, image, label, mask):
        image = cv2.resize(image, self.size,interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size,interpolation=cv2.INTER_NEAREST)
        mask  = cv2.resize(mask,  self.size,interpolation=cv2.INTER_NEAREST)
        return image, label, mask

class ToTensor(object):
    def __call__(self, image, labels, mask):
        return torch.from_numpy(image.astype('float32')).permute(2, 0, 1), \
               torch.from_numpy(labels[None,:,:].astype('float32')), torch.from_numpy(mask[None,:,:].astype('float32'))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, label, mask):
        for t in self.transforms:
            img, label, mask = t(img, label, mask)
        return img, label, mask

class RandomSaturation(object):
    def __init__(self, lower=0.8, upper=1.2):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, labels, mask):
        if random.randint(0, 2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, labels, mask

class RandomContrast(object):
    def __init__(self, lower=0.8, upper=1.2):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, labels, mask):
        if random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, labels, mask

class ConvertFromInts(object):
    def __call__(self, image, labels, mask):
        return image.astype(np.float32), labels, mask

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current
    def __call__(self, image, labels, mask):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, labels, mask

class RandomHue(object):
    def __init__(self, delta=15.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
    def __call__(self, image, labels, mask):
        if random.randint(0, 2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, labels, mask

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, image, label, mask):
        if random.randint(0, 2):
            delta = random.randint(-self.delta, self.delta)
            image += delta
        return image, label, mask

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, image, labels, mask):
        if random.randint(0, 1):
            swap = self.perms[random.randint(0,len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, labels, mask

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(0.9, 1.1),
            ConvertColor(transform='HSV'),
            RandomSaturation(0.9, 1.1),
            RandomHue(10),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(0.9, 1.1)
        ]
        self.rand_brightness = RandomBrightness(10)
        self.rand_light_noise = RandomLightingNoise()
    def __call__(self, image, labels, mask):
        im = image.copy()
        im, labels, mask = self.rand_brightness(im, labels, mask)
        #if random.randint(0, 1):
        #    im = FancyPCA(im)
        if random.randint(0, 1):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, labels, mask = distort(im, labels, mask)
        return im, labels, mask#self.rand_light_noise(im, labels, mask)

class Normalize(object):
    def __call__(self, image, labels, mask):
        image = image.astype(np.float32)
        image /= 255.0
        return image*4., labels, mask

