# -*- coding: utf-8 -*-
import numpy as np

def gen_seperable_points(n, m=0):
    """
    Generate 2n linear seperable points.
    """
    samples1 = np.vstack((-10 * np.random.rand(1, n) + 5, -5 * np.random.rand(1, n) - m))
    samples2 = np.vstack((-10 * np.random.rand(1, n) + 5, 5 * np.random.rand(1, n) + m))
    transform = np.hstack((np.random.randn(2, 2), np.random.randn(2, 1) * 5))
    samples1 = transform.dot(np.vstack((samples1, np.ones(n))))
    samples2 = transform.dot(np.vstack((samples2, np.ones(n))))
    return samples1, samples2
