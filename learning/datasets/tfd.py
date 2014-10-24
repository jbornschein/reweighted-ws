"""

Access to the Totonto Face Dataset 

"""

from __future__ import division

import os
import logging

import numpy as np
from scipy.io import loadmat

import theano
import theano.tensor as T


from learning.datasets import DataSet, datapath

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

#-----------------------------------------------------------------------------

class TorontoFaceDataset(DataSet):
    def __init__(self, which_set='train', size=48, fold=0, n_datapoints=-1, path="TFD", preproc=[]):
        super(TorontoFaceDataset, self).__init__(preproc)

        _logger.info("Loading Toronto Face Dataset (48x48)")

        fname = datapath(path)

        if size == 48:
            fname += "/TFD_48x48.mat"
        elif size == 96:
            fname += "/TFD_96x96.mat"
        else:
            raise ValueError("Unknown size %s. Allowerd options 48 or 96." % size)

        assert 0 <= fold and fold <= 4

        # Load dataset 
        data = loadmat(fname)

        if which_set == 'unlabeled':
            idx = (data['folds'][:,fold] == 0)
        elif which_set == 'train':
            idx = (data['folds'][:,fold] == 1)
        elif which_set == 'unlabeled+train':
            idx =  (data['folds'][:,fold] == 0)
            idx += (data['folds'][:,fold] == 1)
        elif which_set == 'valid':
            idx = (data['folds'][:,fold] == 2)
        elif which_set == 'test':
            idx = (data['folds'][:,fold] == 3)
        else:
            raise ValueError("Unknown dataset %s" % which_set)

        X = data['images'][idx,:,:]
        #Y = data['labs_id'][idx,:]

        if n_datapoints > 0:
            X = X[:n_datapoints]
            Y = Y[:n_datapoints]
        else:
            n_datapoints = X.shape[0]

        # Normalize to 0..1 
        X = (X / 255.).astype(floatX)

        # Flatten images
        X = X.reshape([n_datapoints, -1])

        self.n_datapoints = n_datapoints
        self.X = X
        self.Y = None

