#!/usr/bin/env python 

from __future__ import division

import sys

import abc
import logging
from six import iteritems
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

theano_rng = RandomStreams(seed=2341)
floatX = theano.config.floatX

#=============================================================================
# Trainer base class
class TrainerBase(HyperBase):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **hyper_params):
        super(TrainerBase, self).__init__()

        self.logger = logging.getLogger("trainer")
        self.dlog = dlog.getLogger("trainer")

        self.step = 0 

        self.register_hyper_param("model", default=None, help="")
        self.register_hyper_param("dataset", default=None, help="")
        self.register_hyper_param("termination", default=None, help="")
        self.register_hyper_param("final_monitors", default=[], help="")
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
        self.shvar[name] = theano.shared(value, name=name, allow_downcast=True)
        self.shvar_update_fnc[name] = update_fnc
        
    def update_shvars(self):
        for key, shvar in iteritems(self.shvar):
            value = self.shvar_update_fnc[key](self)
            if isinstance(value, np.ndarray):
                if (value.dtype == np.float32) or (value.dtype == np.float64):
                    value = value.astype(floatX)
            shvar.set_value(value)

    def load_data(self):
        dataset = self.dataset
        assert isinstance(dataset, DataSet)

        n_datapoints = dataset.n_datapoints
        assert n_datapoints == dataset.X.shape[0]

        X, Y = dataset.preproc(dataset.X, dataset.Y)
        self.train_X = theano.shared(X, "train_X")
        self.train_Y = theano.shared(Y, "train_Y")

        self.train_perm = theano.shared(np.random.permutation(n_datapoints))

    def shuffle_train_data(self):
        n_datapoints = self.dataset.n_datapoints
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
        self.register_hyper_param("lr_decay", default=1.0, help="Learning rated decau per epoch")
        self.register_hyper_param("beta", default=0.95, help="Momentum factor")
        self.register_hyper_param("weight_decay", default=0.0, help="Weight decay")
        self.register_hyper_param("batch_size", default=100, help="")
        self.register_hyper_param("sleep_interleave", default=5, help="")
        self.register_hyper_param("layer_discount", default=1.0, help="Reduce LR for each successive layer by this factor")
        self.register_hyper_param("n_samples", default=10, help="No. samples used during training")

        self.mk_shvar('n_samples', 100)
        self.mk_shvar('batch_size', 100)
        self.mk_shvar('permutation', np.zeros(10), lambda self: np.zeros(10))
        self.mk_shvar('beta', 1.0)
        self.mk_shvar('lr_p', np.zeros(2), lambda self: self.calc_learning_rates(self.learning_rate_p))
        self.mk_shvar('lr_q', np.zeros(2), lambda self: self.calc_learning_rates(self.learning_rate_q))
        self.mk_shvar('lr_s', np.zeros(2), lambda self: self.calc_learning_rates(self.learning_rate_s))
        self.mk_shvar('weight_decay', 0.0)

        self.set_hyper_params(hyper_params)
    
    def calc_learning_rates(self, base_rate):
        n_layers = len(self.model.p_layers)
        rng = np.arange(n_layers)
        return base_rate * self.layer_discount ** rng

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
        weight_decay = self.shvar['weight_decay']
        batch_size = self.shvar['batch_size']
        n_samples = self.shvar['n_samples']

        batch_idx = T.iscalar('batch_idx')
        batch_idx.tag.test_value = 0

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[self.train_perm[first:last]]
        #Y_batch = self.train_Y[self.train_perm[first:last]]

        X_batch, _ = self.dataset.late_preproc(X_batch, None)
        
        batch_log_PX, gradients = model.get_gradients(
                    X_batch, None,
                    lr_p=lr_p, lr_q=lr_q,
                    n_samples=n_samples
                )
        batch_log_PX = batch_log_PX / batch_size

        # Initialize momentum variables
        gradients_old = {}
        for shvar, value in iteritems(gradients):
            name = value.name
            gradients_old[shvar] = theano.shared(shvar.get_value()*0., name=("%s_old"%name))

        updates = OrderedDict()
        for shvar, value in iteritems(gradients):
            gradient_old = gradients_old[shvar]

            dTheta = T.switch(T.isnan(value),
                gradient_old,
                beta*gradient_old + (1.-beta)*value
            )

            updates[gradient_old] = dTheta
            updates[shvar] = shvar + dTheta - weight_decay*(shvar+dTheta)

        self.do_step = theano.function(  
                            inputs=[batch_idx],
                            outputs=batch_log_PX, #, Lp, Lq, w],
                            updates=updates,
                            name="do_step")

        #---------------------------------------------------------------------
        self.logger.info("compiling do_sleep_step")
        n_dreams = T.iscalar('n_dreams')
        n_dreams.tag.test_value = 10 

        beta = self.shvar['beta']
        lr_s = self.shvar['lr_s']
        
        log_PX, gradients = model.get_sleep_gradients(lr_s, n_dreams)
        log_PX = T.sum(log_PX)

        updates = OrderedDict()
        for shvar, value in iteritems(gradients):
            gradient_old = gradients_old[shvar]

            dTheta = T.switch(T.isnan(value),
                gradient_old,
                beta*gradient_old + (1.-beta)*value
            )

            updates[gradient_old] = dTheta
            updates[shvar] = shvar + dTheta - weight_decay*(shvar+dTheta)

        self.do_sleep_step = theano.function(  
                            inputs=[n_dreams],
                            outputs=log_PX,
                            updates=updates,
                            name="do_sleep_step")

    def perform_learning(self):
        self.update_shvars()

        termination = self.termination
        model = self.model

        assert isinstance(termination, Termination)
        assert isinstance(model, Model)
        
        # Print information
        n_datapoints = self.dataset.n_datapoints
        n_batches = n_datapoints // self.batch_size

        self.logger.info("Dataset contains %d datapoints in %d mini-batches (%d datapoints per mini-batch)" %
            (n_datapoints, n_batches, self.batch_size))
        self.logger.info("Using %d training samples" % self.n_samples)
        self.logger.info("lr_p=%3.1e, lr_q=%3.1e, lr_s=%3.1e, lr_decay=%5.1e layer_discount=%4.2f" %
            (self.learning_rate_p, self.learning_rate_q, self.learning_rate_s, self.lr_decay, self.layer_discount))

        epoch = 0
        # Perform first epoch
        saved_step_monitors = self.step_monitors
        self.step_monitors = self.first_epoch_step_monitors + self.step_monitors

        for m in self.step_monitors + self.epoch_monitors:
            m.on_init(model)
            m.on_iter(model)
        

        self.logger.info("Starting epoch 0...")
        L = self.perform_epoch()
        self.step_monitors = saved_step_monitors

        # remaining epochs...
        termination.reset()
        while termination.continue_learning(L):
            epoch = epoch + 1
            self.logger.info("Starting epoch %d..." % epoch)
            L = self.perform_epoch()

        # run final_monitors after lerning converged...
        self.logger.info("Calling final_monitors...")
        for m in self.final_monitors:
            m.on_init(model)
            m.on_iter(model)

    #-----------------------------------------------------------------------
    def perform_epoch(self):
        n_datapoints = self.dataset.n_datapoints
        batch_size = self.batch_size
        n_batches = n_datapoints // batch_size
        epoch = self.step // n_batches
        LL_epoch = 0

        self.update_shvars()
        self.shuffle_train_data()

        # Update learning rated
        self.shvar['lr_p'].set_value((self.calc_learning_rates(self.learning_rate_p / self.lr_decay**epoch)).astype(floatX))
        self.shvar['lr_q'].set_value((self.calc_learning_rates(self.learning_rate_q / self.lr_decay**epoch)).astype(floatX))
        self.shvar['lr_s'].set_value((self.calc_learning_rates(self.learning_rate_s / self.lr_decay**epoch)).astype(floatX))

        widgets = ["Epoch %d, step "%(epoch+1), pbar.Counter(), ' (', pbar.Percentage(), ') ', pbar.Bar(), ' ', pbar.Timer(), ' ', pbar.ETA()]
        bar = pbar.ProgressBar(widgets=widgets, maxval=n_batches).start()

        t0 = time()
        while True:
            LL = self.perform_step(update=False)
            LL_epoch += LL

            batch_idx = self.step % n_batches
            bar.update(batch_idx)

            if self.step % n_batches == 0:
                break
        t = time()-t0
        bar.finish()

        LL_epoch /= n_batches

        self.logger.info("Completed epoch %d in %.1fs (%.1fms/step). Calling epoch_monitors..." % (epoch+1, t, t/n_batches*1000))
        for m in self.epoch_monitors:
            m.on_iter(self.model)

        self.dlog.append_all({
            'timing.epoch':  t,
            'timing.step': t/n_batches
        })
        return LL_epoch

    def perform_step(self, update=True):
        n_batches = self.dataset.n_datapoints // self.batch_size
        batch_idx = self.step % n_batches

        # Do we need to update shared variables/parameters?
        if update:
            self.update_shvars()

        LL = self.do_step(batch_idx)

        #
        self.step = self.step + 1
        epoch = self.step // n_batches
        batch_idx = self.step % n_batches

        self.dlog.append("pstep_L", LL)

        if (self.step % self.sleep_interleave == 0) and (self.learning_rate_s > 0.0):
            self.logger.debug("Epoch %d, step %d (%d steps total): Performing sleep cycle\x1b[K" % (epoch+1, batch_idx, self.step))
            n_dreams = self.sleep_interleave * self.batch_size
            sleep_LL = self.do_sleep_step(n_dreams)
        else:
            sleep_LL = np.nan
        self.dlog.append("psleep_L", sleep_LL)

        if (self.step % self.monitor_nth_step == 0) and (len(self.step_monitors) > 0):
            self.logger.info("Epoch %d, step %d (%d steps total): Calling step_monitors...\x1b[K" % (epoch+1, batch_idx, self.step))
            for m in self.step_monitors:
                m.on_iter(self.model)

        return LL
