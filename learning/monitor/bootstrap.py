#!/usr/bin/env python 

from __future__ import division

import abc
import logging

import numpy as np

import theano 
import theano.tensor as T

from learning.dataset import DataSet
from learning.model import Model
from learning.monitor import Monitor
from learning.models.rws import f_replicate_batch, f_logsumexp
import learning.utils.datalog as datalog

from theano.tensor.shared_randomstreams import RandomStreams

floatX = theano.config.floatX
theano_rng = RandomStreams(seed=2341)

_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
def batch_bootstrap(data, bootstrap_size, n_bootstraps, bootstrap_func):
    """
    """
    def scan_func(prev_res, prev_res2, data, bootstrap_size):
        high = data.shape[1]
        idx = theano_rng.random_integers(size=(bootstrap_size,), low=0, high=(high-1))
        data_ = data[:,idx]
        
        res = bootstrap_func(data_)

        # Reduce
        next_res  = prev_res  + T.sum(res)
        next_res2 = prev_res2 + T.sum(res**2)
        return next_res, next_res2
        #return T.sum(res), T.sum(res**2)

    result, updates = theano.scan(fn=scan_func, 
                        outputs_info=[0., 0.],
                        non_sequences=[data, bootstrap_size],
                        n_steps=n_bootstraps, 
                     )
    
    res, res2 = result
    return res[-1], res2[-1]

#-----------------------------------------------------------------------------

class BootstrapLL(Monitor):
    """ Monitor the LL after each training epoch on an arbitrary 
        test or validation data set
    """
    def __init__(self, data, n_samples, n_bootstraps=None, name=None):
        super(BootstrapLL, self).__init__(name)

        assert isinstance(data, DataSet)
        self.dataset = data

        if isinstance(n_samples, int):
            n_samples = [n_samples]

        self.n_samples = n_samples
        self.max_samples = max(n_samples)

        # n_bootstraps
        if n_bootstraps is None:
            n_bootstraps = int(self.max_samples)
        self.n_bootstraps = n_bootstraps

        # max_samples
        if self.max_samples <= 10:
            self.batch_size = 100
        elif self.max_samples <= 100:
            self.batch_size = 10
        else:
            self.batch_size = 1
        

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        p_layers = model.p_layers
        q_layers = model.q_layers
        n_layers = len(p_layers)

        dataset = self.dataset
        X, Y = dataset.preproc(dataset.X, dataset.Y)
        self.X = theano.shared(X, "X")
        self.Y = theano.shared(Y, "Y")

        batch_idx  = T.iscalar('batch_idx')
        n_bootstraps = T.iscalar('n_bootstraps')
        batch_size = self.batch_size
        n_samples = self.n_samples
        max_samples = self.max_samples

        self.logger.info("compiling do_loglikelihood")

        first = batch_idx*batch_size
        last  = first + batch_size

        X_batch, Y_batch = dataset.late_preproc(self.X[first:last], self.Y[first:last])
        X_batch = f_replicate_batch(X_batch, max_samples)
        samples, log_p, log_q = model.sample_q(X_batch)

        # Reshape and sum
        log_p_all = T.zeros((batch_size, max_samples))
        log_q_all = T.zeros((batch_size, max_samples))
        for l in xrange(n_layers):
            samples[l] = samples[l].reshape((batch_size, max_samples, p_layers[l].n_X))
            log_p[l] = log_p[l].reshape((batch_size, max_samples))
            log_q[l] = log_q[l].reshape((batch_size, max_samples))
            log_p_all += log_p[l]   # agregate all layers
            log_q_all += log_q[l]   # agregate all layers
        log_pq = log_p_all - log_q_all

        def bootstrap_func(log_pq):
            # log_pg has shape (batch_size, samples)
            K = log_pq.shape[1]
            #K = 1
            log_px = f_logsumexp(log_pq, axis=1) - T.cast(T.log(K), 'float32')
            return log_px

        outputs = []
        for bootstrap_size in n_samples:
            log_px, log_px2 = batch_bootstrap(log_pq, bootstrap_size, n_bootstraps, bootstrap_func)
            outputs += [log_px, log_px2]

        self.do_loglikelihood = theano.function(  
                            inputs=[batch_idx, n_bootstraps],
                            outputs=outputs,
                            name="do_likelihood")

        #log_PX, _, _, _, KL, Hp, Hq = model.log_likelihood(X_batch, n_samples=n_samples)
        #batch_log_PX = T.sum(log_PX)
        #batch_KL = [T.sum(kl) for kl in KL]
        #batch_Hp = [T.sum(hp) for hp in Hp]
        #batch_Hq = [T.sum(hq) for hq in Hq]

    def on_init(self, model):
        self.compile(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        batch_size = self.batch_size
        n_bootstraps = self.n_bootstraps
        n_datapoints = self.dataset.n_datapoints

        n_layers = len(model.p_layers)

        # Iterate over dataset
        log_px    = [0.] * len(n_samples)
        log_px2   = [0.] * len(n_samples)
        for batch_idx in xrange(n_datapoints//batch_size):
            outputs = self.do_loglikelihood(batch_idx, n_bootstraps)

            for i, K in enumerate(n_samples):
                log_px[i]  += outputs[0]
                log_px2[i] += outputs[1]
                outputs = outputs[2:]

        # Calculate final results and display/store
        for i, K in enumerate(n_samples):
            n    = n_datapoints*n_bootstraps
            LL   = log_px[i] / n
            LLse = np.sqrt( (log_px2[i] - (log_px[i]**2/n)) / (n-1)) 
            LLse *= 1.96 / np.sqrt(n)

            self.logger.info("(%d datpoints, %d samples, %d bootstraps): LL=%5.2f +-%4.2f" % (n_datapoints, K, n_bootstraps, LL, LLse))

            prefix = "spl%d." % K
            self.dlog.append_all({
                prefix+"LL": LL,
                prefix+"LL_se": LLse,
            })

        global validation_LL
        validation_LL = LL
