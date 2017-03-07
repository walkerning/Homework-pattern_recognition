# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import time
from contextlib import contextmanager

@contextmanager
def profile_context(name):
    start_time = time.time()
    sys.stdout.flush()
    yield
    elapsed_time = time.time() - start_time
    print("[PROFILE] {}: {} s elapsed.".format(name, elapsed_time))

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
