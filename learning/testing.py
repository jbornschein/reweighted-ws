
import numpy as np

import theano 
import theano.tensor as T

theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'raise'


def iscalar(name=None):
    Av = 1
    A  = T.iscalar(name=name)
    A.tag.test_value = Av
    return A, Av

def fscalar(name=None):
    Av = 1.23
    A  = T.fscalar(name=name)
    A.tag.test_value = Av
    return A, Av

def ivector(size, name=None):
    Av = np.zeros(size, dtype=np.int)
    A  = T.iscalar(name=name)
    A.tag.test_value = Av
    return A, Av

def fvector(size, name=None):
    Av = np.zeros(size, dtype=np.float32)
    A  = T.fscalar(name=name)
    A.tag.test_value = Av
    return A, Av

def imatrix(shape, name=None):
    Av = np.zeros(shape, dtype=np.int)
    A  = T.imatrix(name=name)
    A.tag.test_value = Av
    return A, Av

def fmatrix(shape, name=None):
    Av = np.zeros(shape, dtype=np.float32)
    A  = T.fmatrix(name=name)
    A.tag.test_value = Av
    return A, Av

