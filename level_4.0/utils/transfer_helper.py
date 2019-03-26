#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/24
author: lujie
"""

import PIL
import numpy as np
import torch as t
from IPython import embed
import matplotlib.pyplot as plt
import torchvision.transforms as T


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img, size=512):
    '''
    Normalize the input image,  unsqueeze demension

    Input:
    - img: (H, W, 3)

    Returns:
    - (1, 3, H, 3)
    '''
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Lambda(rescale),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(), std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img):
    '''
    Denormalize, squeeze the dimension

    Input:
    - (1, 3, H, W) or (3, H, W)

    Returns:
    - (H, W, 3)
    '''

    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    ''' Rescale the image '''

    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def extract_features(x, model):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - model: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """

    features, prev_feat = [], x

    for i, module in enumerate(model._modules.values()):

        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat

    return features


def features_from_img(imgpath, imgsize, model, dtype = t.FloatTensor):
    ''' extract the features from image '''

    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(dtype)
    features = extract_features(img_var, model)
    return features, img_var


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """

    loss = content_weight * t.sum((content_current - content_original) ** 2)

    return loss


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """

    N, C, H, W = features.size()

    features = features.view(N, C, H*W)

    # .matmul: https://pytorch.org/docs/stable/torch.html#torch.matmul
    # .permute: https://discuss.pytorch.org/t/swap-axes-in-pytorch/970
    gram = features.matmul(features.permute(0, 2, 1))

    if normalize:
        gram /= (H * W * C)

    return gram


def style_loss(feats, style_layers, style_targets, style_weights, dtype = t.FloatTensor):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.

    loss = t.tensor(0.).type(dtype)

    for i in range(len(style_layers)):
        loss += style_weights[i] * t.sum((gram_matrix(feats[style_layers[i]]) - style_targets[i]) ** 2)

    return loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!

    H_variation = t.sum((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2)
    W_variation = t.sum((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2)

    return tv_weight * (H_variation + W_variation)


def content_loss_test(refer_out, model):

    imgpath = './styles/tubingen.jpg'
    image_size, dtype = 192, t.FloatTensor
    content_layer, content_weight = 3, 6e-2

    c_feats, content_img_var = features_from_img(imgpath, image_size, model)
    bad_img = t.zeros(*content_img_var.data.size()).type(dtype)
    feats = extract_features(bad_img, model)

    infer_out = content_loss(content_weight, c_feats[content_layer], feats[content_layer]).cpu().data.numpy()

    error = rel_error(refer_out, infer_out)

    print('Maximum error is {:.3f}'.format(error))


def gram_matrix_test(refer_out, model):
    ''' test the gram_matrix '''

    style_image = './styles/starry_night.jpg'
    style_size = 192
    feats, _ = features_from_img(style_image, style_size, model)
    infer_out = gram_matrix(feats[5].clone()).cpu().data.numpy()
    error = rel_error(refer_out, infer_out)
    print('Maximum error is {:.3f}'.format(error))


def style_loss_test(refer_out, model):
    ''' test the style_loss '''

    content_image = './styles/tubingen.jpg'
    style_image = './styles/starry_night.jpg'
    image_size, style_size = 192, 192

    style_layers = [1, 4, 6, 7]
    style_weights = [300000, 1000, 15, 3]

    c_feats, _ = features_from_img(content_image, image_size, model)
    feats, _   = features_from_img(style_image, style_size, model)

    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    infer_out = style_loss(c_feats, style_layers, style_targets, style_weights).cpu().data.numpy()
    error = rel_error(refer_out, infer_out)

    print('Error is {:.3f}'.format(error))


def tv_loss_test(refer_out):
    ''' test the tv '''

    content_image = './styles/tubingen.jpg'
    image_size, tv_weight = 192, 2e-2

    content_img = preprocess(PIL.Image.open(content_image), size=image_size)

    infer_out = tv_loss(content_img, tv_weight).cpu().data.numpy()
    error = rel_error(refer_out, infer_out)
    print('Error is {:.3f}'.format(error))


def style_transfer(model, content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = True, dtype = t.FloatTensor):
    """
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    - dtype: turn the image into floatTensor
    """

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img = content_img.type(dtype)
    content_feats = extract_features(content_img, model)
    content_target = content_feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_img = style_img.type(dtype)
    style_feats = extract_features(style_img, model)
    print(content_img.shape, style_img.shape)

    # for i in range(len(content_feats)):
    #     print(i+1, content_feats[i].shape)
    # print('-' * 100)
    # for i in range(len(style_feats)):
    #     print(i+1, style_feats[i].shape)


    style_targets = []
    for idx in style_layers:
        A = gram_matrix(style_feats[idx].clone())
        # style_targets.append(gram_matrix(style_feats[idx].clone()))
        # print(A.shape)
        style_targets.append(A)
    # Initialize output image to content image or noise
    if init_random:
        img = t.Tensor(content_img.size()).uniform_(0, 1).type(dtype)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img.requires_grad_(True)

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img Torch tensor, whose requires_grad flag is set to True
    optimizer = t.optim.Adam([img], lr=initial_lr)

    plt.ion()
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess(content_img.cpu()))
    axarr[1].imshow(deprocess(style_img.cpu()))
    plt.pause(5)
    plt.ioff()

    for index in range(400):
        if index < 190:
            img.data.clamp_(-1.5, 1.5)

        optimizer.zero_grad()

        feats = extract_features(img, model)

        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(img, tv_weight)
        loss   = c_loss + s_loss + t_loss

        loss.backward()

        # Perform gradient descents on our image values
        if index == decay_lr_at:
            optimizer = t.optim.Adam([img], lr=decayed_lr)
        optimizer.step()

        if index % 200 == 0:
            print('Iteration {}'.format(index))
            plt.ion()
            plt.axis('off')
            plt.imshow(deprocess(img.data.cpu()))
            plt.pause(5)
            plt.ioff()

    print('Iteration {}'.format(t))
    plt.ion()
    plt.axis('off')
    plt.imshow(deprocess(img.data.cpu()))
    plt.pause(5)
    plt.ioff()
