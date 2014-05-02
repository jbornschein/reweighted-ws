#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from model import Model, default_weights
from utils.unrolled_scan import unrolled_scan

floatX = theano.config.floatX

_logger = logging.getLogger(__name__)

theano.config.exception_verbosity = 'high'
theano_rng = MRG_RandomStreams(seed=2341)

class NADE(Model):
    def __init__(self, **hyper_params):
        super(NADE, self).__init__()

        self.register_hyper_param('n_vis', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('clamp_sigmoid', default=False)
        self.register_hyper_param('unroll_scan', default=1)

        self.register_model_param('c', help='hidden bias' , default=lambda: np.zeros(self.n_hid))
        self.register_model_param('b', help='visible bias', default=lambda: np.zeros(self.n_vis))
        self.register_model_param('W', help='encoder weights', default=lambda: default_weights(self.n_vis, self.n_hid) )
        self.register_model_param('V', help='decoder weights', default=lambda: default_weights(self.n_hid, self.n_vis) )
        
        self.set_hyper_params(hyper_params)

    def f_sigmoid(self, x):
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    def f_loglikelihood(self, X):
        n_vis, n_hid = self.get_hyper_params(['n_vis', 'n_hid'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])
        
        batch_size = X.shape[0]
        vis = X

        #------------------------------------------------------------------
        a_init    = T.zeros([batch_size, n_hid], dtype=floatX) + c
        post_init = T.zeros([batch_size], dtype=floatX)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype=floatX)
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0

        return post[-1,:]

    def f_sample(self, n_samples=100):
        n_vis, n_hid = self.get_hyper_params(['n_vis', 'n_hid'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])

        a_init    = T.zeros((n_samples, n_hid), dtype=floatX) + c
        post_init = T.zeros(n_samples, dtype=floatX)
        vis_init  = T.zeros(n_samples, dtype=floatX)
        rand      = theano_rng.uniform((n_vis, n_samples), nstreams=256)

        def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            vis_i = 1.*(rand_i <= pi)
            post  = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype=floatX)
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b, rand], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )
        #assert len(updates) == 0

        return vis.T, post[-1,:]

