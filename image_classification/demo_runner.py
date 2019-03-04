#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/02/17
author: lujie
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from utils.data_utils import load_CIFAR10
from utils.gradient_check import eval_numerical_gradient
from utils.features import extract_features, hog_feature, color_histogram_hsv, pca_feature
from classifier.knn_classifier import KNearestNeighbor
from classifier.linear_classifier import LinearSVM, Softmax
from classifier.nn_classifier import TwoLayerNet

class ClassifierEngine(object):
    def __init__(self, data_name = 'CIFAR10'):
        ''' '''
        self.data_name = data_name
        self.dataset   = {}
        self.num_train = None
        self.num_valid = None
        self.num_test  = None
        self.num_dim   = None
        self.num_class = None

    def __features_transform__(self, nbins = 10):
        ''' gen mid-level features for image '''

        print('extract the features from raw-image, please wait ...')
        feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin = nbins)]
        self.dataset['train_X'] = extract_features(self.dataset['train_X'], feature_fns, verbose = False)
        self.dataset['valid_X'] = extract_features(self.dataset['valid_X'], feature_fns, verbose = False)
        self.dataset['test_X']  = extract_features(self.dataset['test_X'], feature_fns, verbose = False)


        # self.dataset['train_X'] = pca_feature(self.dataset['train_X'])
        # self.dataset['valid_X'] = pca_feature(self.dataset['valid_X'])
        # self.dataset['test_X'] = pca_feature(self.dataset['test_X'])

    def dataloader(self, fea_trans = False, report_details = True):
        ''' load data, not reshape the data, shape : [N, H, W, C] '''

        print('load the dataset, please wait...')
        dataset = {}
        if self.data_name is 'CIFAR10':
            dataset = load_CIFAR10()
            if not dataset:
                raise TypeError('load dataset error, please check root_path')
            self.dataset, self.num_class = dataset, len(set(dataset['test_Y']))
            self.num_train, self.num_valid, self.num_test = dataset['train_X'].shape[0], dataset['valid_X'].shape[0], dataset['test_X'].shape[0]
        else:
            raise TypeError('Unknown dataset, please check...')

        if fea_trans:
            self.__features_transform__(nbins = 10)

        self.num_dim = np.prod(dataset['train_X'].shape[1:])

        if report_details:
            dataset_details = 'num_train : %d; num_valid : %d; num_test : %d; num_dim : %d; num_class : %d' \
                              % (self.num_train, self.num_valid, self.num_test, self.num_dim, self.num_class)
            print(dataset_details)

    def visual_prederrors(self, pred_Y = None):
        ''' visual the instances that classifier makes a mistakes  '''

        examples_per_class = 8
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for cls, cls_name in enumerate(classes):
            idxs = np.where((self.dataset['test_Y'] != cls) & (pred_Y == cls))[0]
            idxs = np.random.choice(idxs, examples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
                plt.imshow(self.dataset['test_X'][idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls_name)
        plt.show()

    def visual_train_process(self, notes = None):
        ''' visual the train process '''

        if not notes:
            raise TypeError('there is nothing in notes, please check ...')
        # visual the loss curve
        plt.subplot(2, 1, 1)
        plt.plot(notes['loss_history'])
        plt.title('Loss_history'); plt.xlabel('Iteration'); plt.ylabel('Loss')

        # visual the acc curve
        plt.subplot(2, 1, 2)
        plt.plot(notes['train_acc_history'], label='train')
        plt.plot(notes['val_acc_history'], label='val')
        plt.title('Classification accuracy history')
        plt.xlabel('Epoch'); plt.ylabel('Clasification accuracy')
        plt.show()

    def ckeck_gradient(self, net = None):
        ''' '''
        loss, grads = net.loss(self.dataset['test_X'].reshape(self.num_test, -1), self.dataset['test_Y'], reg = 1e-5)
        for param_name in grads:
            f = lambda W: (net.loss(self.dataset['test_X'].reshape(self.num_test, -1), self.dataset['test_Y'], reg = 1e-5))[0]
            param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose = False)
            print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

    def knn_classifier(self, K = 10):
        ''' KNearestNeighbor classifier '''

        classifier = KNearestNeighbor()
        classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'])
        pred_Y = classifier.predict(self.dataset['test_X'].reshape(self.num_test, -1), k = K)
        acc = np.mean(self.dataset['test_Y'] == pred_Y)  # 21.50% | 0.4281
        return acc

    def svm_classifier(self, lr = 1e-3, reg_lambda = 2e-5, niters = 1000):
        ''' linear svm classifier '''

        classifier = LinearSVM()
        classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'], \
                             learning_rate = lr, reg = reg_lambda, num_iters = niters)
        pred_Y = classifier.predict(self.dataset['test_X'].reshape(self.num_test, -1))
        self.visual_prederrors(pred_Y)
        acc = np.mean(self.dataset['test_Y'] == pred_Y)  # 27.35% | 0.4525
        return acc

    def softmax_classifier(self, lr = 1e-3, reg_lambda = 1e-5, niters = 1000):
        ''' classifier with softmax loss'''

        classifier = Softmax()
        classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'], \
                             learning_rate = lr, reg = reg_lambda, num_iters = niters)
        pred_Y = classifier.predict(self.dataset['test_X'].reshape(self.num_test, -1))
        acc = np.mean(self.dataset['test_Y'] == pred_Y)  # 28.64% | 0.4182
        return acc

    def neural_networks(self, lr = 1e-3, lr_decay = 0.95, reg_lambda = 1e-5, niter = 8000, nbatch = 128):
        ''' two-layer neural neural networks '''

        classifier = TwoLayerNet(input_size = self.num_dim, hidden_size = 100, output_size = self.num_class)
        train_notes = classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'], \
                                       self.dataset['valid_X'].reshape(self.num_valid, -1), self.dataset['valid_Y'], learning_rate = lr, \
                                       learning_rate_decay = lr_decay, reg = reg_lambda, num_iters = niter, batch_size = nbatch)
        # self.ckeck_gradient(classifier)
        self.visual_train_process(train_notes)
        pred_Y = classifier.predict(self.dataset['test_X'].reshape(self.num_test, -1))
        acc = np.mean(self.dataset['test_Y'] == pred_Y)    # 48%
        return acc

    def classifier_runner(self, classifier_list = None):
        ''' class the classifier to classify the images '''
        acc_result = {}
        if not classifier_list:
            error_info = 'please choose one of following classifiers : 1.knn_classifier 2.svm_classifier 3.softmax_classifier 4.neural_networks'
            raise TypeError(error_info)
        else:
            for classifier in classifier_list:
                if 'knn' in classifier:
                    acc_result['knn_classifier'] = self.knn_classifier(K = 10)
                elif 'svm' in classifier:
                    acc_result['svm_classifier'] = self.svm_classifier(lr = 1e-3, reg_lambda = 2e-5, niters = 1000)
                elif 'softmax' in classifier:
                    acc_result['softmax_classifier'] = self.softmax_classifier(lr = 1e-3, reg_lambda = 2e-5, niters = 1000)
                elif 'neural_networks' in classifier:
                    acc_result['neural_networks'] = self.neural_networks(lr = 1e3, lr_decay = 0.95, reg_lambda = 1e3, niter = 8000, nbatch = 200)
                else:
                    print('there is no %s' % classifier)
        return acc_result

if __name__ == '__main__':
    classifier_engine = ClassifierEngine('CIFAR10')
    classifier_engine.dataloader(fea_trans = False)

    # embed()
    # classifier_list = ['svm_classifier', 'softmax_classifier', 'neural_networks']
    classifier_list = ['neural_networks']
    result_dict = classifier_engine.classifier_runner(classifier_list)

    for classifier, acc in result_dict.items():
        print('%-20s; acc: %6.4f' % (classifier, acc))
