# -*- coding: utf-8 -*-
import numpy as np

from randgen import gen_seperable_points
from perceptron import Perceptron
from matplotlib import pyplot as plt


def main():
    N = 100
    samples1, samples2 = gen_seperable_points(N, 1)
    indexes = np.arange(2 * N)
    np.random.shuffle(indexes)
    labels = np.hstack((np.ones(N), -1 * np.ones(N)))
    samples = np.vstack((samples1.T, samples2.T))
    labels = labels[indexes]
    samples = samples[indexes, :]

    # find sample range
    x_min, y_min = np.min(samples, axis=0)
    x_max, y_max = np.max(samples, axis=0)

    plt.figure(1)
    # plot points
    plt.plot(samples1[0], samples1[1], 'r.')
    plt.plot(samples2[0], samples2[1], 'g.')

    test_margins = [0, 0.1, 5, 10]
    plt_colors = ['r', 'g', 'b', 'y', 'k', 'c']
    for index, margin in enumerate(test_margins):
        percep = Perceptron(margin=margin)
        samples = samples.copy()
        percep.train(samples, labels)
        print "margin : {}; weights: {}".format(margin, percep.weights)
        y_x_min = (- x_min * percep.weights[0] - percep.weights[2]) / percep.weights[1] 
        y_x_max = (- x_max * percep.weights[0] - percep.weights[2]) / percep.weights[1] 
        plt.plot([x_min, x_max], [y_x_min, y_x_max], plt_colors[index],
                 label="margin={}".format(margin))

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
