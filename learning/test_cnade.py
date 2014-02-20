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
    cnade =CNADE()
    assert cnade.batch_size == 100

def test_constructor2():
    model = CNADE(**params)
    assert model.n_vis == 16
    assert model.n_hid == 32
    assert model.n_cond == 8

def test_loglikelihood():
    model = CNADE(**params)

    X = T.fmatrix('X')
    Y = T.fmatrix('Y')

    L = model.f_loglikelihood(X, Y)
    L_total = L.mean()
    
    do_loglikelihood = theano.function([X, Y], L_total, name='loglikelihood')


def test_sample():
    model = CNADE(**params)

    Y = T.fmatrix('Y')
    X = model.f_sample(Y)
    
    do_sample = theano.function([Y], X, name='sample')

    # Now, actual values!
    Y = np.zeros( (model.batch_size, model.n_cond), dtype=np.float32 )
    X = do_sample(Y)

