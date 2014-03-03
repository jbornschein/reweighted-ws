#!/usr/bin/env python 

from __future__ import division

import sys

import logging
from collections import OrderedDict 
from time import time

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print

from utils.datalog import dlog, StoreToH5, TextPrinter
from training import Trainer

_logger = logging.getLogger(__name__)

class TrainSTBP(Trainer):
    def __init__(self, batch_size=100, learning_rate=1., layer_discount=1., 
                    beta=.95, n_samples=100, recalc_LL=() ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.layer_discount = layer_discount
        self.n_samples = n_samples
        self.beta = beta

        self.recalc_LL = recalc_LL
    
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
            self.train_Y = theano.shared(data_train.Y, "train_Y")

        if data_valid is not None:
            self.data_valid = data_valid

        if data_train is not None:
            self.data_test = data_test
    
    def compile(self):
        """ Theano-compile neccessary functions """
        model = self.model
        beta = self.beta 

        model.setup()

        #---------------------------------------------------------------------
        _logger.info("compiling f_sgd_step")

        learning_rate = T.fscalar('learning_rate')
        layer_discount = T.fscalar('layer_discount')
        batch_idx = T.iscalar('batch_idx')
        n_samples = T.iscalar('n_samples')
    
        first = batch_idx*self.batch_size
        last  = first + self.batch_size
        X_batch = self.train_X[first:last]
        #Y_batch = self.train_Y[first:last]
        
        batch_log_PX, gradients = model.get_gradients(X_batch, 
                    learning_rate=learning_rate,
                    layer_discount=layer_discount,
                    n_samples=n_samples)

        # Initialize momentum variables
        gradients_old = {}
        for shvar, value in gradients.iteritems():
            name = value.name
            gradients_old[shvar] = theano.shared(shvar.get_value()*0., name=("%s_old"%name))

        updates = OrderedDict()
        for shvar, value in gradients.iteritems():
            gradient_old = gradients_old[shvar]

            dTheta = beta*gradient_old + (1.-beta)*value

            updates[gradient_old] = dTheta
            updates[shvar] = shvar + dTheta

        self.do_sgd_step = theano.function(  
                            inputs=[batch_idx, n_samples, learning_rate, layer_discount], 
                            outputs=batch_log_PX, #, Lp, Lq, w],
                            updates=updates,
                            name="sgd_step",
                            allow_input_downcast=True,
                            on_unused_input='warn')

        #---------------------------------------------------------------------
        """
        _logger.info("compiling do_sleep_step")
        learning_rate = T.fscalar('learning_rate')
        n_samples = T.iscalar('n_samples')
        
        X, H, lQ = model.f_sleep(n_samples)
        total_Lq = T.sum(lQ)

        updates = OrderedDict()
        for pname in model.Q_params:
            p = model.get_model_param(pname)

            dTheta_old = momentum[p]
            curr_grad = 0.5*learning_rate*T.grad(total_Lq, p)

            dTheta = beta*dTheta_old + (1-beta)*curr_grad

            updates[dTheta_old] = dTheta
            updates[p] = p + dTheta
        
        self.do_sleep_step = theano.function(  
                            inputs=[n_samples, learning_rate], 
                            outputs=total_Lq, 
                            updates=updates,
                            name="sleep_step",
                            allow_input_downcast=True,
                            on_unused_input='warn')
        """

        #---------------------------------------------------------------------
        if len(self.recalc_LL) > 0:
            _logger.info("compiling do_loglikelihood")
            batch_idx  = T.iscalar('batch_idx')
            batch_size = T.iscalar('batch_size')
            n_samples  = T.iscalar('n_samples')
    
            first = batch_idx*batch_size
            last  = first + batch_size
            X_batch = self.train_X[first:last]
        
            log_PX, _, _, _ = model.log_likelihood(X_batch, n_samples=n_samples)
            batch_log_PX = T.sum(log_PX)

            self.do_likelihood = theano.function(  
                            inputs=[batch_idx, batch_size, n_samples], 
                            outputs=batch_log_PX, #, Lp, Lq, w],
                            name="do_likelihood",
                            allow_input_downcast=True,
                            on_unused_input='warn')

        #---------------------------------------------------------------------
        if 'exact' in self.recalc_LL:
            _logger.info("compiling true_loglikelihood")
            Lp_exact = model.f_exact_loglikelihood(self.train_X)
            Lp_exact = T.mean(Lp_exact)

            self.do_exact_LL  = theano.function(
                                inputs=[],
                                outputs=Lp_exact, 
                                name="true_LL",
                                allow_input_downcast=True,
                                on_unused_input='warn')

        #---------------------------------------------------------------------
        _logger.debug("compiling f_loglikelihood")
        
        #n_samples = T.fmatrix('n_samples')
        #X = T.fmatrix('X')
        #
        #Lp, Lq, H, w = model.f_loglikelihood(X)
        #total_LL = T.mean(T.sum(Lp*w, axis=1))
        #
        #self.f_loglikelihood = theano.function(
        #                    inputs=[X], 
        #                    outputs=[total_LL, Lp*w],
        #                    name="f_loglikelihood",
        #                    allow_input_downcast=True)

    def calc_test_LL(self):
        t0 = time()
        n_test = min(5000, self.data_train.n_datapoints)
        batch_size = max(self.batch_size, 10)
        for spl in self.recalc_LL:
            if spl == 'exact':
                Lp_recalc = self.do_exact_LL()
            else:
                Lp_recalc = 0.
                for batch_idx in xrange(n_test // batch_size):
                    Lp = self.do_likelihood(batch_idx, batch_size, spl) 
                    Lp_recalc += Lp
                Lp_recalc  /= n_test
                
            _logger.info("Test LL with %s samples: Lp_%s=%f " % (spl, spl, Lp_recalc))
            dlog.append("Lp_%s"%spl, Lp_recalc)
        t = time()-t0
        _logger.info("Calculating test LL took %f s"%t)

    def perform_epoch(self):
        model = self.model
        n_samples = self.n_samples
        learning_rate = self.learning_rate
        layer_discount = self.layer_discount

        batch_size = self.batch_size
        n_batches  = self.data_train.n_datapoints // batch_size

        t0 = time()
        Lp_epoch = 0
        for batch_idx in xrange(n_batches):
            batch_log_PX = self.do_sgd_step(batch_idx, n_samples, learning_rate=learning_rate, layer_discount=layer_discount)

            assert np.isfinite(batch_log_PX)
            Lp_epoch  += batch_log_PX

            #for name, p in model.get_model_params().iteritems():
            #    assert np.isfinite(p.get_value()).all(), "%s contains NaN or infs" % name

            _logger.debug("SGD step (%4d of %4d)\tLp=%f" % (batch_idx, n_batches, batch_log_PX/batch_size))
        Lp_epoch  /= n_batches*batch_size
        
        _logger.info("LogLikelihoods: Lp=%f \t" % (Lp_epoch))
                        
        t = time()-t0
        _logger.info("Runtime: %5.2f s/epoch; %f ms/(SGD step)" % (t, t/n_batches*1000))
        return Lp_epoch

#    def evaluate_loglikelihood(self, data):
#        total_LL, LL = self.f_loglikelihood(data)
#        return total_LL, LL

