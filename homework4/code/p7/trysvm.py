# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm

samples_w1 = np.array([[-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -3.4, -4.1, -5.1, 1.9],
                       [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1]]).T

samples_w2 = np.array([[-2.0, -8.9, -4.2, -8.5, -6.7, -0.5, -5.3, -8.7, -7.1, -8.0],
                       [-8.4, 0.2, -7.7, -3.2, -4.0, -9.2, -6.7, -6.4, -9.7, -6.3]]).T

def transform_data(data):
    # return 1 x1 x2 x1**2 x2**2 x1x2
    return np.hstack((np.ones((data.shape[0], 1)), data, data**2, (data[:, 0] * data[:, 1])[:, np.newaxis]))

def main():
    # set misclassification penalty to a large enough value
    trans_samples_w1 = transform_data(samples_w1)
    trans_samples_w2 = transform_data(samples_w2)
    # data = np.vstack((trans_samples_w1[0, :], trans_samples_w2[0, :]))
    # labels = [0, 1]
    # res = svm.SVC(C=1e10, kernel="linear").fit(data, labels)
    # m = np.sqrt(res.coef_[0].dot(res.coef_[0]))
    # margin1 = (res.coef_.dot(trans_samples_w1[0,:]) + res.intercept_) / m
    # margin2 = (res.coef_.dot(trans_samples_w2[0,:]) + res.intercept_) / m
    # print "margin of w1 {}: {}; margin of w2 {}: {}".format(trans_samples_w1[0, :], margin1,
    #                                                         trans_samples_w2[0, :], margin2)

    for num in range(1, samples_w1.shape[0]+1):
        data = np.vstack((trans_samples_w1[:num, :], trans_samples_w2[:num, :]))
        labels = np.hstack((np.zeros(num), np.ones(num)))
        res = svm.SVC(C=1e10, kernel="linear").fit(data, labels)
        print "sample number: {}, coef: {}, b: {}, margin: {}".format(num*2, res.coef_, res.intercept_, np.sqrt(1/(res.coef_[0].dot(res.coef_[0]))))

if __name__ == "__main__":
    main()
