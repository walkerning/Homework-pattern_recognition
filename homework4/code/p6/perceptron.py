# -*- coding: utf-8 -*-
import numpy as np

class Perceptron(object):
    MAX_ITER = 100000

    def __init__(self, **config):
        super(Perceptron, self).__init__()
    
        self.margin = config.get("margin", 0)
        self.max_iter = config.get("max_iter", Perceptron.MAX_ITER)
        self.weights = None

    def train(self, samples, labels):
        """
        Parameters
        ----
        samples: np.ndarray nsamples x N
        labels: np.ndarray nsamples x 1
        """
        assert len(samples.shape) == 2
        nsamples, N = samples.shape
        assert len(labels.shape) in {1, 2}
        assert labels.shape[0] == nsamples
        labels = labels.reshape((nsamples, 1))

        # import pdb
        # pdb.set_trace()
        samples = np.hstack((samples, np.ones((nsamples, 1))))
        reg_samples = samples * np.tile(labels, (1, N + 1))
        self.weights = np.zeros(N + 1)
        last_modify_ind = -1
        for k in range(self.max_iter):
            ind = k % nsamples
            if self.weights.dot(reg_samples[ind]) <= self.margin:
                self.weights = self.weights + reg_samples[ind]
                last_modify_ind = ind
            elif last_modify_ind == ind:
                print "seperable after {} iters.".format(k)
                break
        return self.weights

    def test(self, x):
        assert self.weights is not None, "Model not trained"
        return (self.weights.dot(x.T) > 0).astype(np.int) * 2 - 1

