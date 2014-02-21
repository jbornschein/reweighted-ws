import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing

#_mpoorp Unit Under Test
from isb import ISB, f_replicate_batch

params = {
    'n_vis'     : 16, 
    'n_hid'     : 32,
    'n_qhid'    : 16,
    'n_samples' : 10
}

def test_constructor():
    model = ISB(**params)

    assert model.n_samples == 10

def test_replicat():
    Av = np.array( [[1., 2., 3.], 
                    [2., 3., 4.]]).astype(np.float32)
    A = T.fmatrix('A')
    A.tag.test_value = Av

    B = f_replicate_batch(A, 10)

    do_replicate = theano.function([A], B, name="f_replicate", allow_input_downcast=True)
    
    Bv = do_replicate(Av)
    
    assert Bv.shape == (20, 3)
    assert Bv[0 , 0] == 1.
    assert Bv[10, 0] == 2.
    
def test_q_sample():
    model = ISB(**params)

    batch_size = 10
    N = batch_size * model.n_samples

    X, Xv = testing.fmatrix( (N, model.n_vis), "X")
    H, lQ = model.f_q_sample(X)

    do_q_sample = theano.function([X], [H, lQ], name="f_q_sample")

    Hv, lQv = do_q_sample(Xv)

    assert Hv.shape == (N, model.n_hid)
    assert lQv.shape == (N,)

def test_p():
    model = ISB(**params)

    N = 10

    X, Xv = testing.fmatrix( (N, model.n_vis), name="X")
    H, Hv = testing.fmatrix( (N, model.n_hid), name="H")

    pXH = model.f_p(X, H)
    do_p = theano.function([X, H], pXH, name="f_p")

    pXHv = do_p(Xv, Hv)
    assert pXHv.shape == (N,)
    
def test_p_sample():
    model = ISB(**params)

    n_samples = 10

    H, X = model.f_p_sample(n_samples)

    do_p_sample = theano.function([], [H, X], name="f_p_sample")

    Hv, Xv = do_p_sample()

    assert Hv.shape == (n_samples, model.n_hid)
    assert Xv.shape == (n_samples, model.n_vis)


def test_loglikelihood():
    N = 20
    model = ISB(**params)

    X, X_ = testing.fmatrix( (N, model.n_vis), name="X")

    lP, lQ, H, w = model.f_loglikelihood(X)
    do_loglikelihood = theano.function([X], [lP, lQ, H, w], name='loglikelihood')

    lP_, lQ_, H_, w_ = do_loglikelihood(X_)

    print
    print "lP.shape: ", lP_.shape
    print "lP.shape: ", lQ_.shape
    print "H.shape:  ", H_.shape
    print "w.shape:  ", w_.shape


#def test_sample():
#    model = CNADE(**params)
#
#    Y = T.fmatrix('Y')
#    X = model.f_sample(Y)
#    
#    do_sample = theano.function([Y], X, name='sample')
#
#    # Now, actual values!
#    Y = np.zeros( (model.batch_size, model.n_cond), dtype=np.float32 )
#    X = do_sample(Y)
#
