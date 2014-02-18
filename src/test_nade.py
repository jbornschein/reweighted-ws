import unittest 

import numpy as np

import theano 
import theano.tensor as T

# Unit Under Test
from nade import * 



def test_constructor():
    nade = NADE()
    assert nade.batch_size == 100

def test_constructor2():
    nade = NADE(n_vis=16, n_hid=32)
    assert nade.n_vis == 16
    assert nade.n_hid == 32

def test_loglikelihood():
    nade = NADE(n_vis=16, n_hid=32)

    X = T.fmatrix('X')

    LL = nade.f_loglikelihood(X)
    LL_total = LL.mean()
    
    f_loglikelihood = theano.function([X], LL_total, name='loglikelihood')
