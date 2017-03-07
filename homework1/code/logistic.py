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
from learning import LearnController
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
        "epoch": 40,
        "weight_decay": 0.8,
        "base_learning_rate": 1,
        "decay_learning_rate": 0.5,
        "sparse": True,
        "learn_controller": "plateau"
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
        self.base_learning_rate = float(self.config["base_learning_rate"])
        self.decay_learning_rate = float(self.config["decay_learning_rate"])
        # Multiple types of Learning rate controller
        # TODO: pass in more context like accuracy history,
        #       set (global) learning rate accordingly
        # self.learning_rate_controller = lambda epoch: 0.1**epoch

        # FIXME: max epoch judgement and so on should also be moved in to learn controller
        # for **central managment**
        self.learn_controller = LearnController.get_registry(self.config["learn_controller"])(self.max_epoch, self.base_learning_rate, self.decay_learning_rate)

        # TODO: test interval
        #       now default 1 epoch 1 test

        # Maximum word index, one final dimension for bias
        self.x_dim = int(self.config.get("max_word_id", 0)) + 1

        # Working mode: sparse/dense
        # TODO: combine this two mode elegantly in train
        self.sparse = self.config["sparse"]

        # Number of labels
        self.y_dim = len(index2name_dict)
        # TODO: multiple initialization schems
        #self.weight = np.random.normal(scale=0.001, size=(self.y_dim, self.x_dim))
        self.weight = np.zeros(shape=(self.y_dim, self.x_dim))
        print("x_dim: {}; y_dim: {}".format(self.x_dim, self.y_dim))
        print ("Configuration: ", self.config)

    def test_val(self, val_samples):
        if val_samples:
            num_tested, num_error = test_data(self, "val", val_samples)
            error_rate = float(num_error)/num_tested
            self.val_error_rates.append(error_rate)
            logger.info("error rate on **val set**: {} ({}/{})".format(error_rate, num_error, num_tested))

    def train(self, samples, val_samples=[]):
        train_num = len(samples)
        train_indexes = range(train_num)
        self.learning_rate = self.base_learning_rate
        self.val_error_rates = []
        self.test_val(val_samples)
        for epoch in range(self.max_epoch):
            print("Epoch {}/{}:".format(epoch+1, self.max_epoch))
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
                    normalized_sample = self.normalize_sample(s[0])
                    p = self.calculate_p_from_sample(normalized_sample)
                    # x = self.sample2x(s[0])
                    # p = self.calculate_p(x)

                    # construct the gradient
                    g = np.zeros(shape=(self.y_dim,))
                    g[s[1]] = 1
                    g = g - p
                    # import pdb
                    # pdb.set_trace()
                    for word, n_count in normalized_sample:
                        neg_gradient[:, word] = neg_gradient[:, word] + g * n_count

                # update weight using this gradient
                # import pdb
                # pdb.set_trace()
                # neg_gradient -= self.weight_decay * self.weight
                self.weight += neg_gradient * self.learn_controller.learning_rate
                num += step_n
            self.test_val(val_samples)
            cont = self.learn_controller.decide(
                #base_learning_rate=self.base_learning_rate,
                #learning_rate=self.learning_rate,
                #decay_learning_rate=self.decay_learning_rate,
                epoch=epoch,
                new_error_rate=self.val_error_rates[-1]
            )
            if not cont:
                # Indicate the learning can be terminated, not continue any more
                break
            print("")

    def normalize_sample(self, sample):
        sample_total_count = reduce(lambda x, y: x + y[1], sample, 0)
        normalized_sample = [(w, float(c)/sample_total_count) for w, c in sample]
        normalized_sample.append((self.x_dim - 1, 1))
        return normalized_sample

    def calculate_p_from_sample(self, normalized_sample):
        p = np.exp(self.calculate_score_from_sample(normalized_sample))
        p = p / p.sum()
        return p

    def calculate_score_from_sample(self, normalized_sample):
        score = np.zeros(shape=(self.y_dim,))
        for w, n_count in normalized_sample:
            if w >= self.x_dim - 1:
                continue
            for l in range(self.y_dim):
                score[l] += n_count * self.weight[l, w]
        return score

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
        if not self.sparse:
            return self.calculate_score(self.sample2x(sample_x)).argmax()
        else:
            return self.calculate_score_from_sample(self.normalize_sample(sample_x)).argmax()

    @classmethod
    def load(cls, fname):
        self = cls({})
        with open(fname, "r") as f:
            self.weight = np.loadtxt(f)

        self.y_dim, self.x_dim = self.weight.shape
        return self

    def save(self, fname):
        with open(fname, "w") as f:
            np.savetxt(f, self.weight)
