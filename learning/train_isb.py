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

class TrainISB(Trainer):
    def __init__(self, batch_size=100, learning_rate=1., momentum=True, beta=.95, 
                recalc_LL=() ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = False
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

        #---------------------------------------------------------------------
        _logger.info("compiling f_sgd_step")

        learning_rate = T.fscalar('learning_rate')
        batch_idx = T.iscalar('batch_idx')
    
        first = batch_idx*self.batch_size
        last  = first + self.batch_size
        X_batch = self.train_X[first:last]
        Y_batch = self.train_Y[first:last]
        
        Lp, Lq, lPx, lQx, H, w = model.f_loglikelihood(X_batch, Y_batch)

        #w = Print('w')(w)
        total_Lp = T.sum(lPx)
        total_Lq = T.sum(lQx)
        cost_p = T.sum(T.sum(Lp*w, axis=1))
        cost_q = T.sum(T.sum(Lq*w, axis=1))

        updates = OrderedDict()
        for pname in model.P_params:
            p = model.get_model_param(pname)

            dTheta_old = theano.shared(p.get_value()*0., name=("dLp/d%s_old"%pname))
            updates[dTheta_old] = dTheta_old

            curr_grad = learning_rate*T.grad(cost_p, p, consider_constant=[w])

            dTheta = beta*updates[dTheta_old] + (1-beta)*curr_grad

            updates[dTheta_old] = dTheta
            updates[p] = p + dTheta
            
        for pname in model.Q_params:
            p = model.get_model_param(pname)

            dTheta_old = theano.shared(p.get_value()*0., name=("dLq/d%s_old"%pname))
            updates[dTheta_old] = dTheta_old

            curr_grad = learning_rate*T.grad(cost_q, p, consider_constant=[w])

            dTheta = beta*updates[dTheta_old] + (2-beta)*curr_grad

            updates[dTheta_old] = dTheta
            updates[p] = p + dTheta

        self.do_sgd_step = theano.function(  
                            inputs=[batch_idx, learning_rate], 
                            outputs=[total_Lp, total_Lq], #, Lp, Lq, w],
                            updates=updates,
                            name="sgd_step",
                            allow_input_downcast=True,
                            on_unused_input='warn')

        #---------------------------------------------------------------------
        if len(self.recalc_LL) > 0:
            _logger.info("compiling do_loglikelihood")
            batch_size = 100
            n_samples = T.iscalar('n_samples')
            batch_idx = T.iscalar('batch_idx')
    
            first = batch_idx*batch_size
            last  = first + batch_size
            X_batch = self.train_X[first:last]
        
            Lp, Lq, lPx, lQx, H, w = model.f_loglikelihood(X_batch, n_samples=n_samples)
            total_Lp = T.sum(lPx)
            total_Lq = T.sum(lQx)

            self.do_likelihood = theano.function(  
                            inputs=[batch_idx, n_samples], 
                            outputs=[total_Lp, total_Lq], #, Lp, Lq, w],
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
        
        n_samples = T.fmatrix('n_samples')
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

    def perform_epoch(self):
        model = self.model
        learning_rate = self.learning_rate

        batch_size = self.batch_size
        n_batches  = self.data_train.n_datapoints // batch_size

        t0 = time()
        Lp_epoch = 0
        Lq_epoch = 0
        for batch_idx in xrange(n_batches):
            #total_Lp, total_Lq, Lp, Lq, w = self.do_sgd_step(batch_idx, learning_rate)
            total_Lp, total_Lq = self.do_sgd_step(batch_idx, learning_rate)

            #assert np.isfinite(w).all()
            #assert np.isfinite(Lp).all()
            #assert np.isfinite(Lq).all()
            assert np.isfinite(total_Lp)
            assert np.isfinite(total_Lq)
            Lp_epoch  += total_Lp
            Lq_epoch  += total_Lq

            #for name, p in model.get_model_params().iteritems():
            #    assert np.isfinite(p.get_value()).all(), "%s contains NaN or infs" % name

            _logger.debug("SGD step (%4d of %4d)\tLp=%f Lq=%f" % 
                (batch_idx, n_batches, total_Lp/batch_size, total_Lq/batch_size))
            dlog.append("L_step" , total_Lp/batch_size)
            dlog.append("Lq_step", total_Lq/batch_size)
        Lp_epoch  /= n_batches*batch_size
        Lq_epoch  /= n_batches*batch_size
        
        _logger.info("LogLikelihoods: Lp=%f \t \t Lq=%f" % (Lp_epoch, Lq_epoch))

        for spl in self.recalc_LL:
            if spl == 'exact':
                Lp_recalc = self.do_exact_LL()
            else:
                Lp_recalc = 0.
                Lq_recalc = 0.
                for batch_idx in xrange(self.data_train.n_datapoints // 100):
                    Lp, Lq = self.do_likelihood(batch_idx, spl) 
                    Lp_recalc += Lp
                    Lq_recalc += Lq
                Lp_recalc  /= n_batches*batch_size
                Lq_recalc  /= n_batches*batch_size
                
            _logger.info(" with Lp_%s=%f " % (spl, Lp_recalc))
            dlog.append("Lp_%s"%spl, Lp_recalc)
                
                        

        t = time()-t0
        _logger.info("Runtime: %5.2f s/epoch; %f ms/(SGD step)" % (t, t/n_batches*1000))
        return Lp_epoch

#    def evaluate_loglikelihood(self, data):
#        total_LL, LL = self.f_loglikelihood(data)
#        return total_LL, LL

