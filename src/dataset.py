""" 

"""

from __future__ import division

import logging
import cPickle as pickle
import gzip

import numpy as np

_logger = logging.getLogger()

class DataSet(object):
    def _update_N(self):    
        self.train_N = self.train_x.shape[0]
        self.valid_N = self.valid_x.shape[0]
        self.test_N  = self.test_x.shape[0]

        assert self.train_N == self.train_y.shape[0]
        assert self.valid_N == self.valid_y.shape[0]
        assert self.test_N  == self.test_y.shape[0]

class ToyData(DataSet):
    def __init__(self):
        _logger.info("generating toy data")

        x = np.array(
            [[1., 1., 1., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 1., 1., 1.]], dtype='float32')

        l = np.array([[1., 0.], [0., 1.]], dtype='float32')

        self.train_x = np.concatenate([x]*10)
        self.train_y = np.concatenate([l]*10)

        self.valid_x = np.concatenate([x]*2)
        self.valid_y = np.concatenate([l]*2)

        self.test_x = np.concatenate([x]*5)
        self.test_y = np.concatenate([l]*5)
        
        self._update_N()


class MNIST(DataSet):
    def __init__(self, fname="mnist.pkl.gz"):
        _logger.info("loading MNIST data")

        with gzip.open(fname) as f:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(f)

        self.train_x, self.train_y = self.preprocess(train_x, train_y)
        self.valid_x, self.valid_y = self.preprocess(valid_x, valid_y)
        self.test_x , self.test_y  = self.preprocess(test_x, test_y)

        self._update_N()

    def preprocess(self, x, y):
        N = x.shape[0]
        assert N == y.shape[0]
    
        perm = np.random.permutation(N)
        x = x[perm,:]
        y = y[perm,:]

        x = 1.*(x > 0.5)       # binarize x

        one_hot = np.zeros( (N, 10), dtype="float32")
        for n in xrange(N):
            one_hot[n, y[n]] = 1.

        return x.astype('float32'), one_hot.astype('float32')



def permute_cols(x, idx=None):
    if isinstance(x, list) or isinstance(x, tuple):
        if idx is None:
            _, n_vis = x[0].shape
        idx = np.random.permutation(n_vis)
        return [permute(i, idx) for i in x]
    
    if idx is None:
        _, n_vis = x.shape
        idx = np.random.permutation(n_vis)
    return x[:,idx]

       
