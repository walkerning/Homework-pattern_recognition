#! /usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import re
import logging
import argparse

from classifier import Classifier
from common_utils import *

# setup a simple logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("doc-classify")

def handle_doc_input(phase, path="."):
    with open(os.path.join(path, phase + ".label"), "r") as f:
        # todo, do we leave the file open as an iterator to sacrifise speed for memory?
        labels = [int(l.strip()) for l in f]

    features_dict = {}
    max_word_id = 0
    with open(os.path.join(path, phase + ".data"), "r") as f:
        # todo, do we leave the file open as an iterator to sacrifise speed for memory?
        for l in f:
            doc_id, word_id, count = l.split(" ")
            word_id = int(word_id)
            if word_id > max_word_id:
                max_word_id = word_id
            word_id -= 1
            doc_id = int(doc_id)
            features_dict.setdefault(doc_id, []).append((word_id, int(count)))

    training_data_list = [(v, labels[k-1]-1)  for k, v in features_dict.iteritems()]
    index2name_dict = {}
    with open(os.path.join(path, phase + ".map"), "r") as f:
        for l in f:
            name, index = l.split(" ")
            index2name_dict[int(index)] = name
    return training_data_list, index2name_dict, max_word_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="classifier type", choices=Classifier.populate_all_types())
    parser.add_argument("train_path", help="the path of the train data files", default=".")
    parser.add_argument("test_path", help="the path of the test data files", default=".")
    # only `None` and `test` are realized
    parser.add_argument("-v", "--validation", help="use what method for validation. TODO: k-fold",
                        choices=["None", "test"], default="None")
    # not friendly... but who cares...
    parser.add_argument("-c", "--config", help="a string of space-seperated configurations for the classifier. wrong configuration will be discarded. eg. `max_epoch=5 momentum=0.9", default="")
    args = parser.parse_args()

    training_data_list, index2name_dict, max_word_id = handle_doc_input("train", args.train_path)

    config_dict = {}
    for c in re.split("[ \t]+", args.config):
        if not c:
            continue
        c_name, c_value = c.split("=")
        # Note that all of the values is of string type.
        # specific conversion of the configuration values
        # must be done inside specific classifier
        config_dict[c_name] = c_value

    classifier = Classifier.get_classifier_cls(args.type)(index2name_dict,
                                                          max_word_id=max_word_id,
                                                          **config_dict)
    test_data_list, _, _ = handle_doc_input("test", args.test_path)

    with profile_context("training"):
        classifier.train(training_data_list, [] if args.validation == None else test_data_list)
    logger.info("Finished training {} classifier.".format(args.type))

    train_num_tested, train_num_error = test_data(classifier, "test the trainning set", training_data_list)
    train_error_rate = float(train_num_error)/train_num_tested
    logger.info("The error rate on **training set** of {} classifier is {} ({}/{})".format(args.type, train_error_rate, train_num_error, train_num_tested))

    test_num_tested, test_num_error = test_data(classifier, "test the test set", test_data_list)
    test_error_rate = float(test_num_error)/test_num_tested
    logger.info("The error rate on **test set** of {} classifier is {} ({}/{})".format(args.type, test_error_rate, test_num_error, test_num_tested))

if __name__ == "__main__":
    main()
