import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing

#_mpoorp Unit Under Test
from isb import ISB, f_replicate_batch

params = {
    'n_vis'     : 8, 
    'n_hid'     : 8,
    'n_qhid'    : 8,
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
    

def test_ph_sample():
    n_samples = 10
    model = ISB(**params)
    
    H, P = model.f_ph_sample(n_samples)
    do_ph_sample = theano.function([], [H, P], name="f_ph_sample")

    Hv, Pv = do_ph_sample()

    assert Hv.shape == (n_samples, model.n_hid)
    assert Pv.shape == (n_samples,)


def test_p_sample():
    n_samples = 10
    model = ISB(**params)

    H, Hv = testing.fmatrix((n_samples, model.n_hid), 'H')
    X, P = model.f_p_sample(H)
    do_p_sample = theano.function([H], [X, P], name="f_p_sample")

    Xv, Pv = do_p_sample(Hv)

    assert Xv.shape == (n_samples, model.n_vis)
    assert Pv.shape == (n_samples,)


def test_loglikelihood():
    N = 20
    model = ISB(**params)

    X, X_ = testing.fmatrix( (N, model.n_vis), name="X")

    lP, lQ, lPx, lQx, H, w = model.f_loglikelihood(X)
    do_loglikelihood = theano.function([X], [lP, lQ, lPx, lQx, H, w], name='loglikelihood')

    lP_, lQ_, lPx_, lQx, H_, w_ = do_loglikelihood(X_)

    print
    print "lP.shape: ", lP_.shape
    print "lQ.shape: ", lQ_.shape
    print "H.shape:  ", H_.shape
    print "w.shape:  ", w_.shape

def test_true_loglikelihood():
    N = 20
    model = ISB(**params)

    X, Xv = testing.fmatrix( (N, model.n_vis), name="X")

    lP = model.f_exact_loglikelihood(X)
    do_exact_loglikelihood = theano.function([X], [lP], name='true_loglikelihood')

    lPv, = do_exact_loglikelihood(Xv)

    print "lP.shape: ", lPv.shape
    assert lPv.shape == (N,)

