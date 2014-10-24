"""

Access to the MNIST dataset of handwritten digits.

"""

from __future__ import division

import os
import logging
import cPickle as pickle
import gzip

import numpy as np

import theano
import theano.tensor as T

from learning.datasets import DataSet, datapath

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

#-----------------------------------------------------------------------------

class MNIST(DataSet):
    def __init__(self, which_set='train', n_datapoints=None, fname="mnist.pkl.gz", preproc=[]):
        super(MNIST, self).__init__(preproc)

        _logger.info("Loading MNIST data")
        fname = datapath(fname)

        if fname[-3:] == ".gz":
            open_func = gzip.open
        else:
            open_func = open

        with open_func(fname) as f:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(f)

        if which_set == 'train':
            self.X, self.Y = self.prepare(train_x, train_y, n_datapoints)
        elif which_set == 'valid':
            self.X, self.Y = self.prepare(valid_x, valid_y, n_datapoints)
        elif which_set == 'test':
            self.X, self.Y = self.prepare(test_x, test_y, n_datapoints)
        elif which_set == 'salakhutdinov_train':
            train_x = np.concatenate([train_x, valid_x])
            train_y = np.concatenate([train_y, valid_y])
            self.X, self.Y = self.prepare(train_x, train_y, n_datapoints)
        elif which_set == 'salakhutdinov_valid':
            train_x = np.concatenate([train_x, valid_x])[::-1]
            train_y = np.concatenate([train_y, valid_y])[::-1]
            self.X, self.Y = self.prepare(train_x, train_y, n_datapoints)
        else:
            raise ValueError("Unknown dataset %s" % which_set)

        self.n_datapoints = self.X.shape[0]

    def prepare(self, x, y, n_datapoints):
        N = x.shape[0]
        assert N == y.shape[0]

        if n_datapoints is not None:
            N = n_datapoints

        x = x[:N]
        y = y[:N]

        one_hot = np.zeros((N, 10), dtype=floatX)
        for n in xrange(N):
            one_hot[n, y[n]] = 1.

        return x.astype(floatX), one_hot.astype(floatX)

