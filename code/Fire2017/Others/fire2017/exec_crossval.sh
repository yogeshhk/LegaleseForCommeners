#!/bin/sh
../../crf_learn -t -c 10.0 template train_crossval.data model_crossval
../../crf_test  -m model_crossval test_crossval.data > results_crossval.data

