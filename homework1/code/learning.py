# -*- coding: utf-8 -*-

from __future__ import print_function
from registry import Registry

class LearnController(object):
    __metaclass__ = Registry
    REGISTRY_NAME = "LearnController"

    def __init__(self, max_epoch, base_learning_rate, decay_learning_rate):
        self.base_learning_rate = base_learning_rate
        self.decay_learning_rate = decay_learning_rate
        self.max_epoch = max_epoch
        self._learning_rate = self.base_learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    def decide(self, epoch, new_error_rate):
        pass

class PlateauController(LearnController):
    TYPE = "plateau"

    def __init__(self, max_epoch, base_learning_rate, decay_learning_rate):
        super(PlateauController, self).__init__(max_epoch, base_learning_rate, decay_learning_rate)
        self.error_list = []
        self.plateau_threshold = 0.0002
        self.run_threshold = 3
        self.run = 0

    def decide(self, epoch, new_error_rate):
        if epoch >= self.max_epoch:
            return False

        if self.error_list:
            if self.error_list[-1] - new_error_rate < self.plateau_threshold:
                if self.run < self.run_threshold:
                    self._learning_rate *= self.decay_learning_rate
                    print("learning rate ajust to: {}".format(self._learning_rate))
                    self.run += 1
                else:
                    return False
            else:
                self.run = 0

        self.error_list.append(new_error_rate)
        return True

class DecayController(LearnController):
    TYPE = "decay"

    def __init__(self, max_epoch, base_learning_rate, decay_learning_rate):
        super(DecayController, self).__init__(max_epoch, base_learning_rate, decay_learning_rate)

    def decide(self, epoch, new_error_rate):
        if epoch >= self.max_epoch:
            return False

        self._learning_rate *= self.decay_learning_rate
        print("learning rate ajust to: {}".format(self._learning_rate))
        return True
