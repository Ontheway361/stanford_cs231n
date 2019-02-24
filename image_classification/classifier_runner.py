#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2019/02/17
author: lujie
"""
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed
from utils.path_util import DATA_PATH
from utils.data_utils import load_CIFAR10
from utils.features import extract_features, hog_feature, color_histogram_hsv
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

    def dataloader(self, report_details = True):
        ''' load data, not reshape the data, shape : [N, H, W, C] '''

        print('load the dataset, please wait...')
        dataset = {}
        if self.data_name is 'CIFAR10':
            root_path = '%s/cifar-10-batches-py/' % DATA_PATH
            dataset = load_CIFAR10(root_path)
            if not dataset:
                print('load dataset error, please check root_path')
            else:
                self.dataset, self.num_class = dataset, len(set(dataset['test_Y']))
                self.num_train, self.num_dim  = dataset['train_X'].shape[0], np.prod(dataset['train_X'].shape[1:])
                self.num_valid, self.num_test = dataset['valid_X'].shape[0], dataset['test_X'].shape[0]
                if report_details:
                    dataset_details = 'num_train : %d; num_valid : %d; num_test : %d; num_dim : %d; num_class : %d' \
                                      % (self.num_train, self.num_valid, self.num_test, self.num_dim, self.num_class)
                    print(dataset_details)
        else:
            raise TypeError('Unknown dataset, please check...')
        return dataset

    def gen_features(self, nbins = 10):
        ''' gen mid-level features for image '''

        feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin = nbins)]
        self.dataset['train_X'] = extract_features(self.dataset['train_X'], feature_fns, verbose = True)
        self.dataset['valid_X'] = extract_features(self.dataset['valid_X'], feature_fns, verbose = True)
        self.dataset['test_X']  = extract_features(self.dataset['test_X'], feature_fns, verbose = True)
        self.num_dim = np.prod(self.dataset['train_X'].shape[1:])

    def visual_dataset(self):
        ''' visual the dataset '''
        pass

    def knn_classifier(self, K = 10):
        ''' KNearestNeighbor classifier '''

        classifier = KNearestNeighbor()
        classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'])
        pred_Y = classifier.predict(self.dataset['test_X'].reshape(self.num_test, -1), k = K)
        acc = np.mean(self.dataset['test_Y'] == pred_Y)
        return acc

    def svm_classifier(self, lr = 1e-3, reg_lambda = 2e-5, niters = 1000):
        ''' linear svm classifier '''

        classifier = LinearSVM()
        classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'], \
                             learning_rate = lr, reg = reg_lambda, num_iters = niters)
        pred_Y = classifier.predict(self.dataset['test_X'].reshape(self.num_test, -1))
        acc = np.mean(self.dataset['test_Y'] == pred_Y)  # 27.35%
        return acc

    def softmax_classifier(self, lr = 1e-3, reg_lambda = 1e-5, niters = 1000):
        ''' classifier with softmax loss'''

        classifier = Softmax()
        classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'], \
                             learning_rate = lr, reg = reg_lambda, num_iters = niters)
        pred_Y = classifier.predict(self.dataset['test_X'].reshape(self.num_test, -1))
        acc = np.mean(self.dataset['test_Y'] == pred_Y)  # 28.64%
        return acc

    def neural_networks(self, lr = 1e-3, lr_decay = 0.95, reg_lambda = 1e-5, niter = 10000, nbatch = 200):
        ''' two-layer neural neural networks '''

        classifier = TwoLayerNet(input_size = self.num_dim, hidden_size = 200, output_size = self.num_class)
        classifier.train(self.dataset['train_X'].reshape(self.num_train, -1), self.dataset['train_Y'], \
                             self.dataset['valid_X'].reshape(self.num_valid, -1), self.dataset['valid_Y'], \
                             learning_rate = lr, learning_rate_decay = lr_decay, reg = reg_lambda, num_iters = niter, batch_size = nbatch)
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
                    acc_result['neural_networks'] = self.neural_networks(lr = 1e-3, lr_decay = 0.95, reg_lambda = 1e-5, niter = 10000, nbatch = 200)
                else:
                    print('there is no %s' % classifier)
        return acc_result



if __name__ == '__main__':
    classifier_engine = ClassifierEngine('CIFAR10')
    classifier_engine.dataloader()
    classifier_engine.gen_features()

    # classifier_list = ['svm_classifier', 'softmax_classifier', 'neural_networks']
    classifier_list = ['neural_networks']
    result = classifier_engine.classifier_runner(classifier_list)
