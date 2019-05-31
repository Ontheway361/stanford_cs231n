#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/03/25
author: lujie
"""

import torch as t
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from IPython import embed

# dtype = t.FloatTensor
# dtype = t.cuda.FloatTensor

class Flatten(nn.Module):

    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


def show_images(images, frame = 0):
    ''' show the image '''

    images  = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn   = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig= None
    if frame == 0:
        plt.ion()
        fig = plt.figure(figsize=(sqrtn, sqrtn))

    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(sqrtimg, sqrtimg) * 255)
    plt.pause(1)
    plt.ioff()


def preprocess_img(x):
    ''' map the x into [-1, 1] '''
    return 2 * x - 1.0


def deprocess_img(x):
    ''' map the x into [0, 1] '''
    return (x + 1.0) / 2.0


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params(model):
    """ Count the number of parameters in the current TensorFlow graph """

    for param in model.parameters():
        print(param.shape)

    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = (t.rand((batch_size, dim)) - 0.5) * 2
    return noise


def discriminator():
    ''' Build and return a PyTorch model implementing the architecture above '''

    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 256),    # nn.Linear(784, 256), for mnist
        nn.LeakyReLU(0.01),
        nn.Linear(256, 256),     # nn.Linear(256, 256), mnist
        nn.LeakyReLU(0.01),
        nn.Linear(256, 1)        # nn.Linear(256, 1)
    )
    return model


def generator(noise_dim = 96):
    ''' Build and return a PyTorch model implementing the architecture above '''

    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),   # nn.Linear(1024, 784)
        nn.Tanh()
    )
    return model


def dcgan_discriminator(batch_size = 128):
    ''' DCGAN discriminator '''

    classifier = nn.Sequential(
        Unflatten(batch_size, 1, 28, 28),   # Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, kernel_size = 5, stride = 1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(32, 64, kernel_size = 5, stride = 1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        Flatten(),
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),  # nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.LeakyReLU(0.01),
        nn.Linear(4 * 4 * 64, 1)
    )
    return classifier


def dcgan_generator(batch_size = 128, noise_dim = 96):
    ''' DCGAN generator '''

    generator = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024), # Note: nn.BatchNorm1d
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(7 * 7 * 128), # Note: nn.BatchNorm1d
        Unflatten(batch_size, 128, 7, 7),
        nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 1),
        nn.Tanh(),
        Flatten()
    )
    return generator


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """

    loss = - input * target + (1 + input.exp()).log()

    return loss.mean()


def discriminator_loss(logits_real, logits_fake, dtype = t.FloatTensor):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    N = logits_real.size(0)

    labels_real = t.ones(N).type(dtype)
    loss_real = bce_loss(logits_real, labels_real)

    labels_fake = t.zeros(N).type(dtype)
    loss_fake = bce_loss(logits_fake, labels_fake)

    loss = loss_real + loss_fake
    return loss


def generator_loss(logits_fake, dtype = t.FloatTensor):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    labels_fake = t.ones(logits_fake.size(0)).type(dtype)

    loss = bce_loss(logits_fake, labels_fake)

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = 0.5 * t.mean((scores_real - 1) ** 2) + 0.5 * t.mean(scores_fake ** 2)

    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = 0.5 * t.mean((scores_fake - 1) ** 2)

    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

    return optimizer


def gan_runner(loader_train, adversarial = 'affine', loss_type = 'gan', show_every = 250, \
                   batch_size = 128, noise_size = 96, num_epochs = 10, dtype = t.FloatTensor):

    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.

    step - 1. construct the discriminator and generator
    step - 2. get the solver of D and G
    step - 3. set the loss for system
    step - 4. run a gan
    """

    # step - 1
    D, G = None, None
    if adversarial == 'affine':
        D = discriminator().type(dtype)
        G = generator().type(dtype)
    elif adversarial == 'deep_conv':
        D = dcgan_discriminator(128).type(dtype)
        D.apply(initialize_weights)
        G = dcgan_generator(128, 96).type(dtype)
        G.apply(initialize_weights)
    else:
        raise TypeError('unknown adversarial type ...')

    # step - 2
    D_solver, G_solver = get_optimizer(D), get_optimizer(G)

    # step - 3
    discriminate_loss, generate_loss = None, None
    if loss_type == 'gan':
        discriminate_loss = discriminator_loss
        generate_loss = generator_loss
    elif loss_type == 'ls_gan':
        discriminate_loss = ls_discriminator_loss
        generate_loss = ls_generator_loss
    elif loss_type == 'dc_gan':
        discriminate_loss = dc_discriminator_loss
        generate_loss = dc_generator_loss
    else:
        raise TypeError('Unknown loss type...')

    # step - 4
    iter_count = 0
    for epoch in range(num_epochs):

        for x, _ in loader_train:

            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2* (real_data - 0.5)).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminate_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generate_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                show_images(imgs_numpy[0:16], iter_count)
            iter_count += 1
