#!/usr/bin/env python 

from __future__ import division

import sys

import logging
from collections import OrderedDict 
from time import time

import numpy as np

import theano 
import theano.tensor as T

from datalog import dlog, StoreToH5, TextPrinter


_logger = logging.getLogger()

class Trainer(object):
    pass

class BatchedSGD(Trainer):
    def __init__(self, batch_size):  
        self.batch_size = batch_size
    
        self.model = None
        self.data_train = None
        self.data_valid = None
        self.data_test = None
    
    def set_model(self, model):
        self.model = model

    def set_data(self, data_train=None, data_valid=None, data_test=None):
        if data_train is not None:
            self.data_train = data_train
            self.train_X = theano.shared(data_train.X, "train_X")

        if data_valid is not None:
            self.data_valid = data_valid

        if data_train is not None:
            self.data_test = data_test
    
    def compile(self):
        """ Theano-compile neccessary functions """
        model = self.model

        #---------------------------------------------------------------------
        _logger.debug("compiling f_sgd_step")

        learning_rate = T.fscalar('learning_rate')
        batch_idx = T.iscalar('batch_idx')
    
        first = batch_idx*self.batch_size
        last  = first + self.batch_size
        batch = self.train_X[first:last]

        LL = model.loglikelihood(batch)
        total_LL = T.mean(LL)

        updates = OrderedDict()
        for p in model.model_params:  
            updates[p] = p + learning_rate * T.grad(total_LL, p)

        self.f_sgd_step = theano.function(  
                            inputs=[batch_idx, learning_rate], 
                            outputs=total_LL,
                            updates=updates,
                            name="f_sgd_step",
                            allow_input_downcast=True)

        #---------------------------------------------------------------------
        _logger.debug("compiling f_loglikelihood")

        data = T.fmatrix('data')

        LL = model.loglikelihood(data)
        total_LL = T.mean(LL)

        self.f_loglikelihood = theano.function(
                            inputs=[data], 
                            outputs=[total_LL, LL],
                            name="f_loglikelihood",
                            allow_input_downcast=True)

    def perform_epoch(self, learning_rate):
        n_batches = self.data_train.n_datapoints // self.batch_size

        t0 = time()
        for batch_idx in xrange(n_batches):

            LL = self.f_sgd_step(batch_idx, learning_rate)

            _logger.info("SGD step (%4d of %4d)\tLL=%f" % (batch_idx, n_batches, LL))
        t = time()-t0
        
        _logger.info("Time per epoch %f s; %f ms per SGD step" % (t, t/n_batches*1000))

#    def evaluate_loglikelihood(self, data):
#        total_LL, LL = self.f_loglikelihood(data)
#        return total_LL, LL
#
#
