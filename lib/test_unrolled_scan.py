import unittest

import numpy as np

import theano
import theano.tensor as T

# unit under test
from unrolled_scan import *

def test_unroll1():
    i = T.arange(100)
    A = theano.shared(np.random.normal(size=(10,10)))

    def fn1(seq, acc):
        return T.dot(acc, A)

    # Unrolled scan
    outputs, updates = unrolled_scan(fn1, name='fn1',
        sequences=[i], outputs_info=[T.ones_like(A)],
        unroll=1
    )
    f_fn1 = theano.function([], outputs[-1], name='fn1')

def test_last_out_only():
    i = T.arange(100)
    A = theano.shared(np.random.normal(size=(10,10)))

    def fn1(seq, acc):
        return T.dot(acc, A)

    # Normal Theano scan
    outputs, updates = theano.scan(fn1, name='fn1',
        sequences=[i], outputs_info=[T.ones_like(A)]
    )
    f_fn1 = theano.function([], outputs[-1], name='fn1')
    res_normal = f_fn1()

    # Unrolled scan
    outputs, updates = unrolled_scan(fn1, name='fn1',
        sequences=[i], outputs_info=[T.ones_like(A)],
        unroll=10
    )
    f_fn1 = theano.function([], outputs[-1], name='fn1')
    res_unrolled = f_fn1()

    assert np.allclose(res_normal, res_unrolled)

