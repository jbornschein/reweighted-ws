#!/usr/bin/env python 

from __future__ import division

import sys

import abc
import logging
from collections import OrderedDict
from time import time

import numpy as np
import progressbar as pbar

import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import utils.datalog as dlog

from hyperbase import HyperBase
from termination import Termination
from dataset import DataSet
from model import Model

_logger = logging.getLogger(__name__)

theano_rng = RandomStreams(seed=2341)
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
        self.register_hyper_param("monitor_nth_step", default=1, help="")

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

        n_datapoints = data.n_datapoints
        assert n_datapoints == data.X.shape[0]
        #assert n_datapoints == data.Y.shape[0]

        self.train_X = theano.shared(data.X, "train_X")
        self.train_Y = theano.shared(data.Y, "train_Y")
        self.train_perm = theano.shared(np.random.permutation(n_datapoints))

    def shuffle_train_data(self):
        n_datapoints = self.data.n_datapoints
        self.train_perm.set_value(np.random.permutation(n_datapoints))

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
        self.register_hyper_param("learning_rate_s", default=1e-2, help="Learning rate")
        self.register_hyper_param("beta", default=0.95, help="Momentum factor")
        self.register_hyper_param("batch_size", default=100, help="")
        self.register_hyper_param("layer_discount", default=1.0, help="Reduce LR for each successive layer by this factor")
        self.register_hyper_param("n_samples", default=10, help="No. samples used during training")

        def calc_learning_rates(base_rate):
            n_layers = len(self.model.layers)
            rng = np.arange(n_layers)
            return base_rate * self.layer_discount ** rng

        self.mk_shvar('n_samples', 100)
        self.mk_shvar('batch_size', 100)
        self.mk_shvar('permutation', np.zeros(10), lambda self: np.zeros(10))
        self.mk_shvar('beta', 1.0)
        self.mk_shvar('lr_p', np.zeros(2), lambda self: calc_learning_rates(self.learning_rate_p))
        self.mk_shvar('lr_q', np.zeros(2), lambda self: calc_learning_rates(self.learning_rate_q))
        self.mk_shvar('lr_s', np.zeros(2), lambda self: calc_learning_rates(self.learning_rate_s))

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
        lr_q = self.shvar['lr_q']
        beta = self.shvar['beta']
        batch_size = self.shvar['batch_size']
        n_samples = self.shvar['n_samples']

        batch_idx = T.iscalar('batch_idx')

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[self.train_perm[first:last]]
        #Y_batch = self.train_Y[self.train_perm[first:last]]
        
        batch_log_PX, gradients = model.get_gradients(X_batch, None,
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
        self.logger.info("compiling do_sleep_step")
        n_dreams = T.iscalar('n_dreams')

        beta = self.shvar['beta']
        lr_s = self.shvar['lr_s']
        
        log_PX, gradients = model.get_sleep_gradients(lr_s, n_dreams)
        log_PX = T.sum(log_PX)

        updates = OrderedDict()
        for shvar, value in gradients.iteritems():
            gradient_old = gradients_old[shvar]

            dTheta = beta*gradient_old + (1.-beta)*value

            updates[gradient_old] = dTheta
            updates[shvar] = shvar + dTheta

        self.do_sleep_step = theano.function(  
                            inputs=[n_dreams],
                            outputs=log_PX,
                            updates=updates,
                            name="do_sleep_step",
                            allow_input_downcast=True,
                            on_unused_input='warn')

    def perform_step(self, batch_idx, update=True):
        if update:
            self.update_shvars()

        LL = self.do_step(batch_idx)

        if batch_idx % self.n_samples == 0:
            self.logger.debug("Performing sleep cycle %d         " % batch_idx)
            self.perform_sleep()

        if batch_idx % self.monitor_nth_step == 0:
            self.logger.info("SGD step %d, calling step_monitors...      " % batch_idx)
            for m in self.step_monitors:
                m.on_iter(self.model)

        self.dlog.append("pstep_L", LL)
        return LL

    def perform_sleep(self):
        n_dreams = self.n_samples * self.batch_size
        LL = self.do_sleep_step(n_dreams)
        self.dlog.append("psleep_L", LL)
        
    def perform_epoch(self):
        self.update_shvars()
        self.shuffle_train_data()


        n_datapoints = self.data.n_datapoints
        n_batches = n_datapoints // self.batch_size
        LL_epoch = 0

        widgets = ["SGD step ", pbar.Counter(), ' (', pbar.Percentage(), ') ', pbar.Bar(), ' ', pbar.Timer(), ' ', pbar.ETA()]
        bar = pbar.ProgressBar(widgets=widgets, maxval=n_batches)
        bar.start()

        t0 = time()
        for batch_idx in xrange(n_batches):
            LL = self.perform_step(batch_idx, update=False)
            LL_epoch += LL
            bar.update(batch_idx)
        t = time()-t0
        bar.finish()

        self.logger.info("Completed epoch (%d datapoints) in %f s; (%f ms per SGD step)" % (n_datapoints, t, t/n_batches*1000))
        LL_epoch /= n_batches
        
        for m in self.epoch_monitors:
            m.on_iter(self.model)

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
        
        # Print information
        n_datapoints = self.data.n_datapoints
        n_batches = n_datapoints // self.batch_size

        self.logger.info("Dataset contains %d datapoints in %d mini-batches (%d datapoints per mini-batch)" %
            (n_datapoints, n_batches, self.batch_size))
        self.logger.info("Using %d samples, lr_p=%3.1e, lr_q=%3.1e, layer_discount=%4.2f" %
            (self.n_samples, self.learning_rate_p, self.learning_rate_q, self.layer_discount))

        epoch = 0
        # Perform first epoch
        saved_step_monitors = self.step_monitors
        self.step_monitors = self.first_epoch_step_monitors + self.step_monitors

        for m in self.step_monitors + self.epoch_monitors:
            m.on_init(model)

        self.logger.info("Starting epoch 0...")
        L = self.perform_epoch()
        self.step_monitors = saved_step_monitors

        # remaining epochs...
        termination.reset()
        while self.termination.continue_learning(L):
            epoch = epoch + 1
            self.logger.info("Starting epoch %d..." % epoch)
            L = self.perform_epoch()

