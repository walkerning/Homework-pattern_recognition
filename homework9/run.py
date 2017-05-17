# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from sklearn.cross_validation import KFold
from decision_tree import Feature01DecisionTree

N_FOLDS = 10

data = loadmat("./Sogou_data/Sogou_webpage.mat")
labels = data["doclabel"].flatten()
data = data["wordMat"]

#for i, (train_inds, val_inds) in enumerate(KFold(data.shape[0], n_folds=N_FOLDS)):
SEED = 1 # for same indexes across different run
np.random.seed(SEED)
inds = range(data.shape[0])
np.random.shuffle(inds)
val_num = int(len(inds)/N_FOLDS)
train_inds = inds[val_num:]
val_inds = inds[:val_num]

dt = Feature01DecisionTree(classes=range(1, 10), min_samples_split=40, max_depth=None)
train_err = dt.fit(data[train_inds], labels[train_inds])
print("trained, train_accuracy: {}".format(1-train_err))
predict_labels = dt.predict(data[val_inds])
val_accuracy = float(sum(predict_labels == labels[val_inds])) / len(val_inds)
print("val accuracy: {}".format(val_accuracy))
print("node number: {}; depth: {}".format(len(dt.nodes), dt.now_mdepth))
