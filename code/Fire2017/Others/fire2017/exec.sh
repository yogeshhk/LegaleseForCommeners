#!/bin/sh
../../crf_learn -t -c 10.0 template train.data model
../../crf_test  -m model test.data > results.data

