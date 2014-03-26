import unittest 

import numpy as np

import theano 
import theano.tensor as T

# Unit Under Test
from learning.termination import * 

def test_ll_maxepochs():
    termination = LogLikelihoodIncrease(min_increase=0.0, max_epochs=10)

    L = -100.
    epoch = 0 
    while termination.continue_learning(L) and (epoch < 12):
        epoch += 1
        L += 1.

    print "Epochs perfrmed: ", epoch
    assert epoch == 10

