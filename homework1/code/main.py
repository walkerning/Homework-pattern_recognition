#! /usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import time
import logging
import argparse
from contextlib import contextmanager

from classifier import Classifier

# setup a simple logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("doc-classify")

def handle_doc_input(phase, path="."):
    with open(os.path.join(path, phase + ".label"), "r") as f:
        # todo, do we leave the file open as an iterator to sacrifise speed for memory?
        labels = [int(l.strip()) for l in f]

    features_dict = {}
    with open(os.path.join(path, phase + ".data"), "r") as f:
        # todo, do we leave the file open as an iterator to sacrifise speed for memory?
        for l in f:
            doc_id, word_id, count = l.split(" ")
            doc_id = int(doc_id)
            features_dict.setdefault(doc_id, []).append((int(word_id), int(count)))

    training_data_list = [(v, labels[k-1]-1)  for k, v in features_dict.iteritems()]
    index2name_dict = {}
    with open(os.path.join(path, phase + ".map"), "r") as f:
        for l in f:
            name, index = l.split(" ")
            index2name_dict[int(index)] = name

    return training_data_list, index2name_dict

@contextmanager
def profile_context(name):
    start_time = time.time()
    print("{} ... ".format(name), end="")
    sys.stdout.flush()
    yield
    elapsed_time = time.time() - start_time
    print("{} s elapsed.".format(elapsed_time))

def test_data(classifier, name, data_list):
    num_tested = 0
    num_error = 0
    with profile_context(name):
        for test_case in data_list:
            test_res = classifier.test(test_case[0])
            num_tested += 1
            if test_res != test_case[1]:
                num_error += 1
    return num_tested, num_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="classifier type", choices=Classifier.populate_all_types())
    parser.add_argument("train_path", help="the path of the train data files", default=".")
    parser.add_argument("test_path", help="the path of the test data files", default=".")
    args = parser.parse_args()

    training_data_list, index2name_dict = handle_doc_input("train", args.train_path)

    classifier = Classifier.get_classifier_cls(args.type)(index2name_dict)
    with profile_context("training"):
        classifier.train(training_data_list)
    logger.info("Finished training {} classifier.".format(args.type))

    test_data_list, _ = handle_doc_input("test", args.test_path)

    train_num_tested, train_num_error = test_data(classifier, "test the trainning set", training_data_list)
    train_error_rate = float(train_num_error)/train_num_tested
    logger.info("The error rate on **training set** of {} classifier is {} ({}/{})".format(args.type, train_error_rate, train_num_error, train_num_tested))

    test_num_tested, test_num_error = test_data(classifier, "test the test set", test_data_list)
    test_error_rate = float(test_num_error)/test_num_tested
    logger.info("The error rate on **test set** of {} classifier is {} ({}/{})".format(args.type, test_error_rate, test_num_error, test_num_tested))

if __name__ == "__main__":
    main()
