import unittest 

import numpy as np

import theano 
import theano.tensor as T

# Unit Under Test
from cnade import * 

params = {
    'n_vis':  16, 
    'n_hid':  32,
    'n_cond': 8
}

def test_constructor():
    model = CNADE(**params)
    assert model.n_vis == 16
    assert model.n_hid == 32
    assert model.n_cond == 8

def test_loglikelihood():
    batch_size = 10
    model = CNADE(**params)

    X_ = np.zeros( (batch_size, model.n_vis),  dtype=np.float32)
    Y_ = np.zeros( (batch_size, model.n_cond), dtype=np.float32)

    X = T.fmatrix('X')
    Y = T.fmatrix('Y')
    X.tag.test_value = X_
    Y.tag.test_value = Y_

    L = model.f_loglikelihood(X, Y)
    
    do_loglikelihood = theano.function([X, Y], L, name='loglikelihood')

    L_ = do_loglikelihood(X_, Y_)
   
    assert L_.shape == (batch_size,)

def test_sample():
    batch_size = 10
    model = CNADE(**params)

    Y_ = np.zeros( (batch_size, model.n_cond), dtype=np.float32 )

    Y = T.fmatrix('Y')
    Y.tag.test_value = Y_

    X = model.f_sample(Y)
    
    do_sample = theano.function([Y], X, name='sample')

    # Now, actual values!
    X_, P_ = do_sample(Y_)

    assert X_.shape == (batch_size, model.n_vis)
    assert P_.shape == (batch_size,)

