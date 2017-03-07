# -*- coding: utf-8 -*-

import math
import copy

from registry import Registry

try:
    import numpy as np
    _sum = np.sum
    _argmax = np.argmax
except ImportError:
    _sum = sum
    _argmax = lambda x: max(enumerate(x), key=lambda y: y[1])[0]

class Classifier(object):
    __metaclass__ = Registry
    REGISTRY_NAME = "classifier"

    @classmethod
    def populate_all_types(cls):
        return cls.list_registry()

    @classmethod
    def get_classifier_cls(cls, typ):
        return cls.get_registry(typ)

    def index2label(self, index):
        pass

    def train(self, samples):
        """
        samples is a iterator or iteratable of training data (x, y)
        x is the feature vector, y is the label index.
        """

    def test(self, x):
        """
        input a feature vector, classify into labels
        """

class DocumentClassifier(Classifier):
    def __init__(self, index2name_dict):
        self.index2name_dict = index2name_dict
        self.num_labels = len(self.index2name_dict)

    def index2label(self, index):
        return self.index2name_dict[index]

class NaiveBayesClassifier(DocumentClassifier):
    TYPE = "NaiveBayes"

    def __init__(self, index2name_dict):
        super(NaiveBayesClassifier, self).__init__(index2name_dict)

        self.count_dict = {l: {} for l in range(self.num_labels)}
        self.prior_count_list = [0 for _ in range(self.num_labels)]
        self.nonexist_log_list = [0 for _ in range(self.num_labels)]

    def train(self, samples):
        max_word_index = 0 # assume there are `max_word_index` types of words
        for sample in samples:
            self.prior_count_list[sample[1]] += 1
            for word, count in sample[0]:
                if word > max_word_index:
                    max_word_index = word
                if word not in self.count_dict[sample[1]]:
                    self.count_dict[sample[1]][word] = 0
                self.count_dict[sample[1]][word] += count

        # normalize the count dict, and record the log of the probability for numerical
        # precision
        for l in self.count_dict.iterkeys():
            # if with numpy dependency, can use np.sum to accelerate
            total = _sum(self.count_dict[l].values())
            total += max_word_index # add one smooth
            log_total = math.log(float(total))
            self.nonexist_log_list[l] = -log_total
            for w in self.count_dict[l]:
                self.count_dict[l][w] = math.log(float(self.count_dict[l][w] + 1)) - log_total

        # normalize the prior count of different document labels
        total_prior = _sum(self.prior_count_list)
        log_total_prior = math.log(float(total_prior))
        # FIXME: do we need add-one smooth for prior?
        # what if there is no training data for some kind of existing label.
        # (this situation do not exists in this dataset)
        self.prior_count_list = [math.log(float(c)) - log_total_prior for c in self.prior_count_list]

    def test(self, x):
        post_list = copy.deepcopy(self.prior_count_list)
        for word, count in x:
            for l in range(len(post_list)):
                logp = self.count_dict[l].get(word, self.nonexist_log_list[l])
                post_list[l] += count * logp

        return _argmax(post_list)

class LogisticClassifier(DocumentClassifier):
    """The multinomial discriminative logistic regression classifier."""
    TYPE = "Logistic"
