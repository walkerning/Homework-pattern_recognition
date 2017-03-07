#-*- coding: utf-8 -*-

from __future__ import print_function
import sys
import copy
import math
import random
import logging
# fixme: do we really need to consider when there are no numpy
#        and write some fallback matrix multiply utility...
import numpy as np

from classifier import DocumentClassifier
from common_utils import test_data

logger = logging.getLogger("logistic")

class LogisticClassifier(DocumentClassifier):
    """
    The multinomial discriminative logistic regression classifier.
    Also called Softmax Regression."""

    TYPE = "Logistic"

    default_config = {
        "batch_size": 0,
        "momentum": 0,
        "epoch": 1,
        "weight_decay": 0.8,
        "learning_rate": 1,
        "decay_learning_rate": 0.5
    }

    def __init__(self, index2name_dict, **config):
        super(LogisticClassifier, self).__init__(index2name_dict)

        self.config = copy.deepcopy(self.default_config)
        self.config.update(config)
        # SGD is not used by default(batch_size = 0) unless a batch size is specified.
        self.batch_size = int(self.config["batch_size"])
        # No momentum is used by default.
        self.momentum = float(self.config["momentum"])
        # Maximum epoch
        self.max_epoch = int(self.config["epoch"])
        # Regularization
        # TODO: multiple types of regularization
        self.weight_decay = float(self.config["weight_decay"])
        # Learning rate
        self.base_learning_rate = float(self.config["learning_rate"])
        self.decay_learning_rate = float(self.config["decay_learning_rate"])
        # Multiple types of Learning rate controller
        # TODO: pass in more context like accuracy history,
        #       set (global) learning rate accordingly
        # self.learning_rate_controller = lambda epoch: 0.1**epoch
        # Maximum word index, one final dimension for bias

        # TODO: test interval
        #       now default 1 epoch 1 test
        self.x_dim = int(self.config.get("max_word_id")) + 1
        # Number of labels
        self.y_dim = len(index2name_dict)
        # FIXME: multiple initialization schems
        self.weight = np.random.normal(scale=0.001, size=(self.y_dim, self.x_dim))
        #self.weight = np.zeros(shape=(self.y_dim, self.x_dim))
        print("x_dim: {}; y_dim: {}".format(self.x_dim, self.y_dim))
        print ("Configuration: ", self.config)

    def train(self, samples, test_samples=[]):
        train_num = len(samples)
        train_indexes = range(train_num)
        for epoch in range(self.max_epoch):
            print("Epoch #{}/{}:".format(epoch+1, self.max_epoch))
            if test_samples:
                num_tested, num_error = test_data(self, "val", test_samples)
                error_rate = float(num_error)/num_tested
                logger.info("epoch #{} begin: error rate on **val set**: {} ({}/{})".format(epoch+1, error_rate, num_error, num_tested))
            if self.batch_size != 0:
                random.shuffle(train_indexes)
                step_n = self.batch_size
            else:
                step_n = train_num
            num = 0
            neg_gradient = np.zeros(shape=(self.y_dim, self.x_dim))
            batch_num_per_epoch = int(math.ceil(float(train_num)/step_n))
            batch_ind = 1
            while num < train_num:
                print("batch {}/{} ...".format(batch_ind, batch_num_per_epoch), end="\r")
                sys.stdout.flush()
                batch_ind += 1
                neg_gradient = neg_gradient * self.momentum
                for index in train_indexes[num:min(train_num, num+step_n)]:
                    s = samples[index]
                    x = self.sample2x(s[0])
                    p = self.calculate_p(x)

                    # construct the gradient
                    g = np.zeros(shape=(self.y_dim, 1))
                    g[s[1]] = 1
                    g = g - p
                    # import pdb
                    # pdb.set_trace()
                    neg_gradient += g.dot(x.transpose())

                # update weight using this gradient
                # import pdb
                # pdb.set_trace()
                # neg_gradient -= self.weight_decay * self.weight
                self.weight += neg_gradient * (self.base_learning_rate * self.decay_learning_rate**(epoch/2))
                num += step_n
            print("")

    def calculate_score(self, x):
        return self.weight.dot(x)

    def calculate_p(self, x):
        unnormalized_p = np.exp(self.weight.dot(x))
        norm_p = unnormalized_p / unnormalized_p.sum()
        return norm_p

    def sample2x(self, sample):
        x = np.zeros(shape=(self.x_dim-1, 1))
        for s in sample:
            if s[0] >= self.x_dim -1:
                continue
            x[s[0]] = s[1]
        x = x / x.sum()
        x = np.vstack((x, [1]))
        return x

    def test(self, sample_x):
        return self.calculate_score(self.sample2x(sample_x)).argmax()
