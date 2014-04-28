"""
"""

from __future__ import division

import abc
import logging
import cPickle as pickle
import gzip
import h5py

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

theano_rng = RandomStreams(seed=2341)


#-----------------------------------------------------------------------------
# Dataset base class
class DataSet(object):
    __metaclass__ = abc.ABCMeta

    def preprocess(self, X, Y):
        """ Given a mini-batch of datapoints in a Theano tensor, this
            method returns a Theano tensor with the preprocessed datapoints
        """
        return X, Y


#-----------------------------------------------------------------------------
class ToyData(DataSet):
    def __init__(self, which_set='train'):
        _logger.info("generating toy data")

        self.which_set = which_set

        X = np.array(
            [[1., 1., 1., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 1., 1., 1.]], dtype=floatX)
        Y = np.array([[1., 0.], [0., 1.]], dtype=floatX)

        if which_set == 'train':
            self.X = np.concatenate([X]*10)
            self.Y = np.concatenate([Y]*10)
        elif which_set == 'valid':
            self.X = np.concatenate([X]*2)
            self.Y = np.concatenate([Y]*2)
        elif which_set == 'test':
            self.X = np.concatenate([X]*2)
            self.Y = np.concatenate([Y]*2)
        else:
            raise ValueError("Unknown dataset %s" % which_set)

        self.n_datapoints = self.X.shape[0]


#-----------------------------------------------------------------------------
class BarsData(DataSet):
    def __init__(self, which_set='train', n_datapoints=1000, D=5):
        _logger.debug("Generating bars data")

        n_vis = D**2
        n_hid = 2*D
        bar_prob = 1./n_hid

        X = np.zeros((n_datapoints, D, D), dtype=floatX)
        Y = 1. * (np.random.uniform(size=(n_datapoints, n_hid)) < bar_prob)

        for n in xrange(n_datapoints):
            for d in xrange(D):
                if Y[n, d] > 0.5:
                    X[n, d, :] = 1.0
                if Y[n, D+d] > 0.5:
                    X[n, :, d] = 1.0

        self.X = X.reshape((n_datapoints, n_vis)).astype(floatX)
        self.Y = Y.astype(floatX)
        self.n_datapoints = n_datapoints


#-----------------------------------------------------------------------------
class MNIST(DataSet):
    def __init__(self, which_set='train', n_datapoints=None, fname="mnist.pkl.gz"):
        _logger.info("Loading MNIST data")

        with gzip.open(fname) as f:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(f)

        if which_set == 'train':
            self.X, self.Y = self.static_preprocess(train_x, train_y, n_datapoints)
        elif which_set == 'valid':
            self.X, self.Y = self.static_preprocess(valid_x, valid_y, n_datapoints)
        elif which_set == 'test':
            self.X, self.Y = self.static_preprocess(test_x, test_y, n_datapoints)
        elif which_set == 'salakhutdinov_train':
            train_x = np.concatenate([train_x, valid_x])
            train_y = np.concatenate([train_y, valid_y])
            self.X, self.Y = self.static_preprocess(train_x, train_y, n_datapoints)
        elif which_set == 'salakhutdinov_valid':
            train_x = np.concatenate([train_x, valid_x])[::-1]
            train_y = np.concatenate([train_y, valid_y])[::-1]
            self.X, self.Y = self.static_preprocess(train_x, train_y, n_datapoints)
        else:
            raise ValueError("Unknown dataset %s" % which_set)

        self.n_datapoints = self.X.shape[0]

    def static_preprocess(self, x, y, n_datapoints):
        N = x.shape[0]
        assert N == y.shape[0]

        if n_datapoints is not None:
            N = n_datapoints

        x = x[:N]
        y = y[:N]

        #perm = np.random.permutation(N)
        #x = x[perm,:]
        #y = y[perm]

        one_hot = np.zeros((N, 10), dtype=floatX)
        for n in xrange(N):
            one_hot[n, y[n]] = 1.

        return x.astype(floatX), one_hot.astype(floatX)

    def preprocess(self, X, Y):
        """ Given a mini-batch of datapoints in a Theano tensor, this
        method returns a Theano tensor with the preprocessed datapoints
        """
        #draw uniform random
        U = theano_rng.uniform(size=X.shape, ndim=2, low=0.1, high=0.9)

        return 1.*(X >= U), Y


#-----------------------------------------------------------------------------
class FromModel(DataSet):
    def __init__(self, model, n_datapoints):
        batch_size = 100

        # Compile a Theano function to draw samples from the model
        n_samples = T.iscalar('n_samples')
        n_samples.tag.test_value = 10

        X, _ = model.sample_p(n_samples)

        do_sample = theano.function(
            inputs=[n_samples],
            outputs=X[0],
            name='sample_p')

        model.setup()
        n_vis = model.n_lower
        #n_hid = model.n_hid

        X = np.empty((n_datapoints, n_vis), dtype=floatX)
        #Y = np.empty((n_datapoints, n_hid), dtype=np.floatX)

        for b in xrange(n_datapoints//batch_size):
            first = b*batch_size
            last = first + batch_size
            X[first:last] = do_sample(batch_size)
        remain = n_datapoints % batch_size
        if remain > 0:
            X[last:] = do_sample(remain)

        self.n_datapoints = n_datapoints
        self.X = X
        self.Y = None

#-----------------------------------------------------------------------------
class FromH5(DataSet):
    def __init__(self, fname, n_datapoints=None, offset=0, table_X="X", table_Y="Y"):
        """ Load a dataset from an HDF5 file.
        """
        with h5py.File(fname, "r") as h5:
            # 
            if not table_X in h5.keys():
                _logger.error("H5 file %s does not contain a table named %s" % (fname, table_X))
                raise ArgumentError()
            
            N_total, D = h5[table_X].shape
            if n_datapoints is None:
                n_datapoints = N_total-offset

            X = h5[table_X][offset:(offset+n_datapoints)]
            if table_Y in h5.keys():
                Y = h5[table_Y][offset:(offset+n_datapoints)]
            else:
                Y = None

        self.X = X.astype(floatX)
        self.Y = Y.astype(floatX)
        self.n_datapoints = self.X.shape[0]

        
#-----------------------------------------------------------------------------
def permute_cols(x, idx=None):
    if isinstance(x, list) or isinstance(x, tuple):
        if idx is None:
            _, n_vis = x[0].shape
        idx = np.random.permutation(n_vis)
        return [permute(i, idx) for i in x]

    if idx is None:
        _, n_vis = x.shape
        idx = np.random.permutation(n_vis)
    return x[:, idx]


#-----------------------------------------------------------------------------
def get_toy_data():
    return BarsData(which_set="train", n_datapoints=500)
