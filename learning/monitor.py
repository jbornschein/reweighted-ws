#!/usr/bin/env python 

from __future__ import division

import abc
import logging

import numpy as np

import theano 
import theano.tensor as T

from dataset import DataSet
from model import Model
from hyperbase import HyperBase
import utils.datalog as datalog

_logger = logging.getLogger(__name__)

class Monitor(HyperBase):
    """ Abtract base class to monitor stuff """

    __metaclass__ = abc.ABCMeta

    def __init__(self):   
        """ 
        """
        self.dlog = datalog.getLogger(__name__)
        self.logger = logging.getLogger(__name__)

    def compile(self):
        pass

    def on_init(self, model):
        """ Called after the model has been initialized; directly before 
            the first learning epoch will be performed 
        """
        self.on_iter(model)

    @abc.abstractmethod
    def on_iter(self, model):
        """ Called whenever a full training epoch has been performed
        """
        pass

class DLogHyperParams(Monitor):
    def __init__(self):
        super(DLogHyperParams, self).__init__()

    def on_iter(self, model):
        model.model_params_to_dlog(self.dlog)

class DLogModelParams(Monitor):
    """
    Write all model parameters to a DataLogger called "model_params".
    """
    def __init__(self):
        super(DLogModelParams, self).__init__()

    def on_iter(self, model):
        model.model_params_to_dlog(self.dlog)

class MonitorLL(Monitor):
    """ Monitor the LL after each training epoch on an arbitrary 
        test or validation data set
    """
    def __init__(self, data, n_samples):
        super(MonitorLL, self).__init__()

        assert isinstance(data, DataSet)
        self.data = data

        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        data = self.data
        assert isinstance(data, DataSet)
        self.train_X = theano.shared(data.X, "train_X")
        self.train_Y = theano.shared(data.Y, "train_Y")

        batch_idx  = T.iscalar('batch_idx')
        batch_size = T.iscalar('batch_size')

        self.logger.info("compiling do_loglikelihood")
        n_samples = T.iscalar("n_samples")
        batch_idx = T.iscalar("batch_idx")
        batch_size = T.iscalar("batch_size")

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[first:last]
        
        log_PX, _, _, _ = model.log_likelihood(X_batch, n_samples=n_samples)
        batch_log_PX = T.sum(log_PX)

        self.do_loglikelihood = theano.function(  
                            inputs=[batch_idx, batch_size, n_samples], 
                            outputs=batch_log_PX, #, Lp, Lq, w],
                            name="do_likelihood",
                            allow_input_downcast=True,
                            on_unused_input='warn')

    def on_init(self, model):
        self.compile(model)
        self.on_iter(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_datapoints = self.data.n_datapoints

        #
        for K in n_samples:
            if K <= 10:
                batch_size = 100
            elif K <= 100:
                batch_size = 10
            else:
                batch_size = 1
    

            # Iterate over dataset
            L = 0
            for batch_idx in xrange(n_datapoints//batch_size):
                log_PX = self.do_loglikelihood(batch_idx, batch_size, K)
                L += log_PX
            L /= n_datapoints

            self.logger.info("Test LL for %d datapoints with %s samples: %d" % (n_datapoints, K, L))
            self.dlog.append("LL_%d"%K, L)
        
        
class SampleFromP(Monitor):
    """ Draw a number of samples from the P-Model """
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def on_iter(self, model):
        raise NotImplemented()

