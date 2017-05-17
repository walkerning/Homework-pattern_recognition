# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from Queue import Queue
from functools import partial

def label_nums(classes, labels):
    assert set(labels).issubset(set(classes))
    return np.array([np.sum(labels == cls) for cls in classes])

def neg_entropy(classes, labels):
    nums = label_nums(classes, labels).astype(np.float) / len(labels)
    nonzero_nums = nums[nums.nonzero()]
    return np.sum(nonzero_nums * np.log(nonzero_nums))

class _DecisionTreeNode(object):
    def __init__(self, objects, labels, classes, depth, criterion_func, purity_func=neg_entropy):
        # criterion will return 0/1 for each feature
        self.objects = objects
        self.labels = labels
        self.classes = classes
        self.criterion_func = criterion_func
        self.depth = depth
        self.left = None
        self.right = None
        self.purity_func = partial(purity_func, classes)

    def decide(self, feature):
        return self.criterion_func(feature)

    def purity(self):
        return self.purity_func(self.labels)

    def majority_label(self):
        nums = label_nums(self.classes, self.labels)
        return self.classes[np.argmax(nums)]

class _Feature01DecisionTreeNode(_DecisionTreeNode):
    def __init__(self, objects, labels, classes, depth, purity_func=neg_entropy):
        def criterion_func(feature):
            if feature.ndim == 1:
                return feature[self.index]
            return feature[:, self.index]
        super(_Feature01DecisionTreeNode, self).__init__(objects, labels, classes, depth, criterion_func, purity_func)

    def decide_split(self):
        if len(self.objects) == 0:
            return None
        max_purity = None
        max_purity_index = 0
        for ind in range(self.objects[0].size):
            lefts = np.where(self.objects[:, ind] == 0)[0]
            rights = np.where(self.objects[:, ind] == 1)[0]
            ratio_lefts = float(len(lefts)) / len(self.objects)
            ratio_rights = 1 - ratio_lefts
            purity = ratio_lefts * self.purity_func(self.labels[lefts]) + ratio_rights * self.purity_func(self.labels[rights])
            if max_purity is None or purity > max_purity:
                max_purity_index = ind
                max_purity = purity
        self.index = max_purity_index
        return self.childrens()

    def childrens(self):
        decides = self.decide(self.objects)
        lefts = np.where(decides == 0)[0]
        rights = np.where(decides == 1)[0]
        return lefts, rights


class Feature01DecisionTree(object):
    def __init__(self, classes, min_samples_split=10, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.nodes = []
        self.classes = classes
        self.now_mdepth = 0 # acutal max depth

    def fit(self, X, Y):
        eps = 1e-20
        # measure the purity by entropy
        q = Queue()
        root_node = _Feature01DecisionTreeNode(X, Y, self.classes, 1)
        q.put(0)
        self.nodes = [root_node]
        err_num = 0
        self.now_mdepth = 1
        while not q.empty():
            node_ind = q.get()
            node = self.nodes[node_ind]
            if node.purity() >= -eps or len(node.objects) <= self.min_samples_split or (self.max_depth is not None and node.depth == self.max_depth):
                # leaf node
                err_num += np.sum(node.labels != node.majority_label())
                if node.depth > self.now_mdepth:
                    self.now_mdepth = node.depth
                continue
            lefts, rights = node.decide_split()
            if len(lefts) == 0 or len(rights) == 0:
                err_num += np.sum(node.labels != node.majority_label())
                if node.depth > self.now_mdepth:
                    self.now_mdepth = node.depth
                continue
            print("depth {} node {}: left num {}, right num {}".format(node.depth, node_ind, len(lefts), len(rights)))
            left_node = _Feature01DecisionTreeNode(node.objects[lefts], node.labels[lefts], self.classes, node.depth + 1)
            right_node = _Feature01DecisionTreeNode(node.objects[rights], node.labels[rights], self.classes, node.depth + 1)
            node.left = left_node
            node.right = right_node
            q.put(len(self.nodes))
            q.put(len(self.nodes)+1)
            self.nodes.append(left_node)
            self.nodes.append(right_node)
        return float(err_num)/X.shape[0]

    def predict(self, X):
        assert len(self.nodes) > 0
        labels = []
        for feature in X:
            node = self.nodes[0]
            while 1:
                if node.left is None:
                    labels.append(node.majority_label())
                    break
                if node.decide(feature) == 0:
                    node = node.left
                else:
                    node = node.right
        return labels
