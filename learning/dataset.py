"""
"""

from __future__ import division

import os
import abc
import logging
import cPickle as pickle
import os.path as path
import gzip
import h5py

import os.path
import numpy as np

import theano
import theano.tensor as T

from preproc import Preproc

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

#-----------------------------------------------------------------------------
def datapath(fname):
    """ Try to find *fname* in the dataset directory and return 
        a absolute path.
    """
    candidates = [
        path.abspath(path.join(path.dirname(__file__), "../data")),
        path.abspath("."),
        path.abspath("data"),
    ]
    if 'DATASET_PATH' in os.environ:
        candidates.append(os.environ['DATASET_PATH'])

    for c in candidates:
        c = path.join(c, fname)
        if path.exists(c):
            return c

    raise IOError("Could not find %s" % fname)

#-----------------------------------------------------------------------------
# Dataset base class
class DataSet(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, preproc=[]):
        self._preprocessors = []
        self.add_preproc(preproc)

    def add_preproc(self, preproc):
        """ Add the given preprocessors to the list of preprocessors to be used

        Parameters
        ----------
        preproc : {Preproc, list of Preprocessors}
        """
        if isinstance(preproc, Preproc):
            preproc = [preproc,]

        for p in preproc:
            assert isinstance(p, Preproc)

        self._preprocessors += preproc
        

    def preproc(self, X, Y):
        """ Statically preprocess data.
        
        Parameters
        ----------
        X, Y : ndarray

        Returns
        -------
        X, Y : ndarray
        """
        for p in self._preprocessors:
            X, Y = p.preproc(X, Y)
        return X, Y

    def late_preproc(self, X, Y):
        """ Preprocess a batch of data
        
        Parameters
        ----------
        X, Y : theano.tensor

        Returns
        -------
        X, Y : theano.tensor
        """
        for p in self._preprocessors:
            X, Y = p.late_preproc(X, Y)
        return X, Y

#-----------------------------------------------------------------------------
class ToyData(DataSet):
    def __init__(self, which_set='train', preproc=[]):
        super(ToyData, self).__init__(preproc)

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
    def __init__(self, which_set='train', n_datapoints=1000, D=5, preproc=[]):
        super(BarsData, self).__init__(preproc)

        n_vis = D**2
        n_hid = 2*D
        bar_prob = 1./n_hid

        X = np.zeros((n_datapoints, D, D), dtype=floatX)
        Y = (np.random.uniform(size=(n_datapoints, n_hid)) < bar_prob).astype(floatX)

        for n in xrange(n_datapoints):
            for d in xrange(D):
                if Y[n, d] > 0.5:
                    X[n, d, :] = 1.0
                if Y[n, D+d] > 0.5:
                    X[n, :, d] = 1.0

        self.X = X.reshape((n_datapoints, n_vis))
        self.Y = Y
        self.n_datapoints = n_datapoints


#-----------------------------------------------------------------------------
class MNIST(DataSet):
    def __init__(self, which_set='train', n_datapoints=None, fname="mnist.pkl.gz", preproc=[]):
        super(MNIST, self).__init__(preproc)

        _logger.info("Loading MNIST data")
        fname = datapath(fname)

        #with gzip.open(fname) as f:
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

        #perm = np.random.permutation(N)
        #x = x[perm,:]
        #y = y[perm]

        one_hot = np.zeros((N, 10), dtype=floatX)
        for n in xrange(N):
            one_hot[n, y[n]] = 1.

        return x.astype(floatX), one_hot.astype(floatX)

#-----------------------------------------------------------------------------
class CalTechSilhouettes(DataSet):
    def __init__(self, which_set='train', n_datapoints=-1, path="caltech-silhouettes", preproc=[]):
        super(CalTechSilhouettes, self).__init__(preproc)

        _logger.info("Loading CalTech 101 Silhouettes data (28x28)")
        path = datapath(path)

        test_x = np.load(path+"/test_data.npy")
        test_y = np.load(path+"/test_labels.npy")

        if which_set == 'train':
            self.X = np.load(path+"/train_data.npy")
            self.Y = np.load(path+"/train_labels.npy")
        elif which_set == 'valid':
            self.X = np.load(path+"/val_data.npy")
            self.Y = np.load(path+"/val_labels.npy")
        elif which_set == 'test':
            self.X = np.load(path+"/test_data.npy")
            self.Y = np.load(path+"/test_labels.npy")
        else:
            raise ValueError("Unknown dataset %s" % which_set)

        self.X = self.X[:n_datapoints]
        self.Y = self.Y[:n_datapoints]

        self.n_datapoints = self.X.shape[0]


#-----------------------------------------------------------------------------
class FromModel(DataSet):
    def __init__(self, model, n_datapoints, preproc=[]):
        super(FromModel, self).__init__(preproc)

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
        n_vis = model.n_X
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
        """ Load a dataset from an HDF5 file. """
        super(FromH5, self).__init__()        

        if not os.path.exists(fname):
            fname = datapath(fname)

        with h5py.File(fname, "r") as h5:
            # 
            if not table_X in h5.keys():
                _logger.error("H5 file %s does not contain a table named %s" % (fname, table_X))
                raise ArgumentError()
            
            N_total, D = h5[table_X].shape
            if n_datapoints is None:
                n_datapoints = N_total-offset

            X = h5[table_X][offset:(offset+n_datapoints)]
            X = X.astype(floatX)
            if table_Y in h5.keys():
                Y = h5[table_Y][offset:(offset+n_datapoints)]
                Y = Y.astype(floatX)
            else:
                Y = None
                Y = X[:,0]

        self.X = X
        self.Y = Y
        self.n_datapoints = self.X.shape[0]


#-----------------------------------------------------------------------------
def get_toy_data():
    return BarsData(which_set="train", n_datapoints=500)
