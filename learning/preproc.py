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

import utils.datalog as dlog

_logger = logging.getLogger(__name__)


floatX = theano.config.floatX

theano_rng = RandomStreams(seed=2341)

#-----------------------------------------------------------------------------
# Base class for preprocessors
class Preproc(object):
    __metaclass__ = abc.ABCMeta

    def preproc(self, X, Y):
        """ Preprocess data and return and X, Y tuple.

        Parameters
        ----------
        X, Y : ndarray

        Returns
        -------
        X, Y : ndarray
        """
        return X, Y

    def late_preproc(self, X, Y):
        """ Preprocess data and return and X, Y tuple.
        
        Parameters
        ----------
        X, Y : theano.tensor
    
        Returns 
        -------
        X, Y : theano.tensor
        """
        return X, Y


#-----------------------------------------------------------------------------
class Binarize(Preproc):
    def __init__(self, threshold=None, late=True):  
        """
        Binarize data in X; assuming that input data was 0 <= X <= 1.

        Parameters
        ----------
        threshold : {float, None}
            Threshold when a input is considerd 1; If None, the value in 
            X determines the probability of the binarized element being a 1.

        late : bool 
            Should binaization be performed statically or each time the data 
            is used?
        """
        if late and not threshold is None:  
            _logger.warning("Using a static threshold and late preprocessing does not make much sense; forcing late=False")
            late = False

        self.threshold = threshold
        self.late = late

    def preproc(self, X, Y):
        """ Binarize X """
        if self.late:    # skip static processing when late == True
            return X, Y
        
        assert (Y >= 0.0).all()
        assert (X <= 1.0).all()

        threshold = self.threshold
        if threshold is None:
            threshold = np.random.uniform(size=X.shape)
        
        X = (X >= threshold).astype(floatX)
        return X, Y
        
    def late_preproc(self, X, Y):
        """ Binarize X """
        if not self.late:
            return X, Y

        threshold = theano_rng.uniform(size=X.shape, ndim=2, low=0.0, high=1.0)
        X = (X >= threshold)
        return X, Y

#-----------------------------------------------------------------------------
class PermuteColumns(Preproc):
    def __init__(self):  
        """
        Create a random permutation and permute each feature-vector X of each 
        datapoint with it.
        """
        self.dlog = dlog.getLogger("preproc.permute_columns")
        self.permutation = None

    def set_permutation(self, permutation):
        self.permutation = permutation

        self.dlog.append("permutation", self.permutation)
        self.dlog.append("permutation_inv", np.argsort(self.permutation))

    def preproc(self, X, Y):
        """ Permute X """
        _, n_vis = X.shape

        if self.permutation is None:
            permutation = np.random.permutation(n_vis)
            self.set_permutation(permutation)

        assert self.permutation.size == n_vis

        X = X[:, self.permutation]

        return X, Y
