#!/usr/bin/env python 

from __future__ import division

import sys

import abc
import logging
from collections import OrderedDict
from time import time

import numpy as np

import theano 
import theano.tensor as T

import utils.datalog as dlog

from hyperbase import HyperBase
from termination import Termination
from dataset import DataSet
from model import Model

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

class TrainerBase(HyperBase):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **hyper_params):
        super(TrainerBase, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.dlog = dlog.getLogger(__name__)

        self.register_hyper_param("model", default=None, help="")
        self.register_hyper_param("data", default=None, help="")
        self.register_hyper_param("termination", default=None, help="")
        self.register_hyper_param("epoch_monitors", default=[], help="")
        self.register_hyper_param("step_monitors", default=[], help="")
        self.register_hyper_param("first_epoch_step_monitors", default=[], help="")

        self.shvar = {}
        self.shvar_update_fnc = {}

        self.set_hyper_params(hyper_params)

    def mk_shvar(self, name, init, update_fnc=None):
        if update_fnc is None:
            update_fnc = lambda self: self.get_hyper_param(name)
        value = init
        if isinstance(value, np.ndarray):
            if (value.dtype == np.float32) or (value.dtype == np.float64):
                value = value.astype(floatX)
        elif isinstance(value, float):
            value = np.asarray(value, dtype=floatX)
        elif isinstance(value, int):
            pass
        else:
            raise ArgumentError('Unknown datatype')
        self.shvar[name] = theano.shared(value, name=name)
        self.shvar_update_fnc[name] = update_fnc
        
    def update_shvars(self):
        for key, shvar in self.shvar.iteritems():
            value = self.shvar_update_fnc[key](self)
            if isinstance(value, np.ndarray):
                if (value.dtype == np.float32) or (value.dtype == np.float64):
                    value = value.astype(floatX)
            shvar.set_value(value)

    def load_data(self):
        data = self.data
        assert isinstance(data, DataSet)
        self.train_X = theano.shared(data.X, "train_X")
        self.train_Y = theano.shared(data.Y, "train_Y")

    @abc.abstractmethod
    def compile(self):
        pass

#=============================================================================
# BatchedSGD trainer
class Trainer(TrainerBase):
    def __init__(self, **hyper_params):
        super(Trainer, self).__init__()

        self.register_hyper_param("learning_rate_p", default=1e-2, help="Learning rate")
        self.register_hyper_param("learning_rate_q", default=1e-2, help="Learning rate")
        self.register_hyper_param("beta", default=0.95, help="Momentum factor")
        self.register_hyper_param("batch_size", default=100, help="")
        self.register_hyper_param("layer_discount", default=1.0, help="Reduce LR for each successive layer by this factor")
        self.register_hyper_param("n_samples", default=10, help="No. samples used during training")
        self.register_hyper_param("recalc_LL", default=())

        def calc_learning_rates(base_rate):
            n_layers = len(self.model.layers)
            rng = np.arange(n_layers)
            return base_rate * self.layer_discount ** rng

        self.mk_shvar('n_samples', 100)
        self.mk_shvar('batch_size', 100)
        self.mk_shvar('beta', 1.0)
        self.mk_shvar('lr_p', np.zeros(2), lambda self: calc_learning_rates(self.learning_rate_p))
        self.mk_shvar('lr_p', np.zeros(2), lambda self: calc_learning_rates(self.learning_rate_q))

        self.mk_shvar('batch_idx', 1, lambda self: 0)

        self.set_hyper_params(hyper_params)
    
    def compile(self):
        """ Theano-compile neccessary functions """
        model = self.model

        assert isinstance(model, Model)

        model.setup()
        self.update_shvars()

        #---------------------------------------------------------------------
        self.logger.info("compiling do_step")

        lr_p = self.shvar['lr_p']
        lr_q = self.shvar['lr_p']
        beta = self.shvar['beta']
        batch_size = self.shvar['batch_size']
        n_samples = self.shvar['n_samples']

        batch_idx = T.iscalar('batch_idx')

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[first:last]
        #Y_batch = self.train_Y[first:last]
        
        batch_log_PX, gradients = model.get_gradients(X_batch, 
                    lr_p=lr_p, lr_q=lr_q,
                    n_samples=n_samples)
        batch_log_PX = batch_log_PX / batch_size

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

        self.do_step = theano.function(  
                            inputs=[batch_idx],
                            outputs=batch_log_PX, #, Lp, Lq, w],
                            updates=updates,
                            name="sgd_step",
                            allow_input_downcast=True,
                            on_unused_input='warn')

        #---------------------------------------------------------------------
        #self.logger.debug("compiling f_loglikelihood")

        #X = T.fmatrix('X')
        #Y = T.fmatrix('Y')

        #LL = model.f_loglikelihood(X, Y)
        #total_LL = T.mean(LL)

        #self.f_loglikelihood = theano.function(
        #                    inputs=[X, Y], 
        #                    outputs=[total_LL, LL],
        #                    name="f_loglikelihood",
        #                    allow_input_downcast=True)


        #---------------------------------------------------------------------
        """
        self.logger.info("compiling do_sleep_step")
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
                
            self.logger.info("Test LL with %s samples: Lp_%s=%f " % (spl, spl, Lp_recalc))
            dlog.append("Lp_%s"%spl, Lp_recalc)
        t = time()-t0
        self.logger.info("Calculating test LL took %f s"%t)

    def perform_epoch(self):
        model = self.model
        n_samples = self.n_samples
        learning_rate = self.learning_rate
        layer_discount = self.layer_discount

        batch_size = self.batch_size
        n_batches  = self.data.n_datapoints // batch_size

        t0 = time()
        Lp_epoch = 0
        for batch_idx in xrange(n_batches):
            batch_log_PX = self.do_sgd_step(batch_idx, n_samples, learning_rate=learning_rate, layer_discount=layer_discount)

            assert np.isfinite(batch_log_PX)
            Lp_epoch  += batch_log_PX

            #for name, p in model.get_model_params().iteritems():
            #    assert np.isfinite(p.get_value()).all(), "%s contains NaN or infs" % name

            self.logger.debug("SGD step (%4d of %4d)\tLp=%f" % (batch_idx, n_batches, batch_log_PX/batch_size))
        Lp_epoch  /= n_batches*batch_size
        
        self.logger.info("LogLikelihoods: Lp=%f \t" % (Lp_epoch))
                        
        t = time()-t0
        self.logger.info("Runtime: %5.2f s/epoch; %f ms/(SGD step)" % (t, t/n_batches*1000))
        return Lp_epoch
    """


#    def evaluate_loglikelihood(self, data):
#        total_LL, LL = self.f_loglikelihood(data)
#        return total_LL, LL

    def perform_step(self, batch_idx, update=True):
        if update:
            self.update_shvars()

        LL = self.do_step(batch_idx)

        for m in self.step_monitors:
            m.on_iter(self.model)

        self.dlog.append("pstep_L", LL)
        return LL

    def perform_sleep(self):
        pass

    def perform_epoch(self):
        self.update_shvars()

        n_batches = self.data.n_datapoints // self.batch_size
        t0 = time()
        LL_epoch = 0
        for batch_idx in xrange(n_batches):
            LL = self.perform_step(batch_idx, update=False)
            LL_epoch += LL

            self.logger.info("SGD step (%4d of %4d)\tLL=%f" % (batch_idx, n_batches, LL))

        t = time()-t0
        LL_epoch /= n_batches
        
        for m in self.epoch_monitors:
            m.on_iter(self.model)

        self.logger.info("Time per epoch %f s; %f ms per SGD step" % (t, t/n_batches*1000))
        self.dlog.append_all({
            'timing.epoch':  t,
            'timing.step': t/n_batches
        })
        return LL_epoch

    def perform_learning(self):
        self.update_shvars()

        termination = self.termination
        model = self.model

        assert isinstance(termination, Termination)
        assert isinstance(model, Model)
        
        epoch = 0
        # Perform first epoch
        saved_step_monitors = self.step_monitors
        self.step_monitors = self.first_epoch_step_monitors + self.step_monitors

        for m in self.step_monitors + self.epoch_monitors:
            m.on_init(model)

        L = self.perform_epoch()
        self.step_monitors = saved_step_monitors

        # remaining epochs...
        while self.termination.continue_learning(L):
            epoch = epoch + 1
            L = self.perform_epoch()

