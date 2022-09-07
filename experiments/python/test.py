#!/usr/bin/env python

# from future import absolute_import, division, print_function

import os
import numpy as np
# import pathlib as pl
from sklearn import linear_model
from scipy import signal

from datasets import caltech, sharee, incart, ucr
import misc_algorithms as algo
import window

A = np.array([[1 + 1j, 2 + 2j]])

A_hat = np.array([[3 + 3j, 4 + 5j]])

diffs = A - A_hat

print(diffs)

print(diffs * diffs)

raw_mse = np.mean(diffs * diffs)

print(raw_mse)

normalized_mse = raw_mse / np.var(A)

print(normalized_mse)