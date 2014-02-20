import unittest 

import numpy as np

import theano 
import theano.tensor as T

# Unit Under Test
from nade import * 

params = {
    'n_vis':  16, 
    'n_hid':  32,
}

def test_constructor():
    model =NADE()
    assert model.batch_size == 100

def test_constructor2():
    model = NADE(**params)
    assert model.n_vis == 16
    assert model.n_hid == 32

def test_loglikelihood():
    model = NADE(**params)

    X = T.fmatrix('X')

    L = model.f_loglikelihood(X)
    L_total = L.mean()
    
    do_loglikelihood = theano.function([X], L_total, name='loglikelihood')


#@unittest.skip("NOT IMPLEMENTED")
def test_sample():
    model = NADE(**params)

    X = model.f_sample()
    
    do_sample = theano.function([], X, name='sample')

    # Now, actual values!
    X = do_sample()

