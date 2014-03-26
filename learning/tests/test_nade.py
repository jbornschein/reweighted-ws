import unittest 

import numpy as np

import theano 
import theano.tensor as T

# Unit Under Test
from learning.nade import * 

params = {
    'n_vis':  16, 
    'n_hid':  32,
}

def test_constructor():
    model = NADE(**params)
    assert model.n_vis == 16
    assert model.n_hid == 32

def test_loglikelihood():
    batch_size = 100
    model = NADE(**params)

    X = T.fmatrix('X')
    X.tag.test_value = np.zeros( (batch_size, model.n_vis), dtype='float32')

    L = model.f_loglikelihood(X)
    
    do_loglikelihood = theano.function([X], L, name='loglikelihood')

    X_ = np.zeros( (batch_size, model.n_vis), dtype=np.float32)
    L_ = do_loglikelihood(X_)
   
    assert L_.shape == (batch_size,)


#@unittest.skip("NOT IMPLEMENTED")
def test_sample():
    n_samples = 10
    model = NADE(**params)

    X, P = model.f_sample(n_samples)
    do_sample = theano.function([], [X, P], name='sample')

    # Now, actual values!
    X_, P_ = do_sample()

    assert X_.shape == (n_samples, model.n_vis)
    assert P_.shape == (n_samples,)

