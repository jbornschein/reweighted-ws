"""

Access to various CalTech datasets.

"""

from __future__ import division

import os
import logging

import numpy as np

import theano
import theano.tensor as T

from learning.datasets import DataSet, datapath

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

#-----------------------------------------------------------------------------

class CalTechSilhouettes(DataSet):
    def __init__(self, which_set='train', n_datapoints=-1, path="caltech-silhouettes", preproc=[]):
        super(CalTechSilhouettes, self).__init__(preproc)

        _logger.info("Loading CalTech 101 Silhouettes data (28x28)")
        path = datapath(path)

        test_x = np.load(path+"/test_data.npy")
        test_y = np.load(path+"/test_labels.npy")

        if which_set == 'train':
            X = np.load(path+"/train_data.npy")
            Y = np.load(path+"/train_labels.npy")
        elif which_set == 'valid':
            X = np.load(path+"/val_data.npy")
            Y = np.load(path+"/val_labels.npy")
        elif which_set == 'test':
            X = np.load(path+"/test_data.npy")
            Y = np.load(path+"/test_labels.npy")
        else:
            raise ValueError("Unknown dataset %s" % which_set)

        if n_datapoints > 0:
            X = X[:n_datapoints]
            Y = Y[:n_datapoints]    
        else:
            n_datapoints = X.shape[0]

        X = X.astype(floatX)

        self.n_datapoints = n_datapoints
        self.X = X
        self.Y = Y

