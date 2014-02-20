#!/usr/bin/env python 

from __future__ import division

import sys
sys.path.insert(0, "../lib")

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import Model, default_weights
from unrolled_scan import unrolled_scan

_logger = logging.getLogger(__name__)

#theano.config.compute_test_value = 'warn'
theano.config.exception_verbosity = 'high'
theano_rng = RandomStreams(seed=2341)

class NADE(Model):
    def __init__(self, **hyper_params):
        super(NADE, self).__init__()

        self.register_hyper_param('n_vis', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('clamp_sigmoid', default=False)
        self.register_hyper_param('batch_size', default=100)
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
        n_vis, n_hid, batch_size = self.get_hyper_params(['n_vis', 'n_hid', 'batch_size'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])
        
        vis = X
        vis.tag.test_value = np.zeros( (batch_size, n_vis), dtype='float32')

        learning_rate = T.fscalar('learning_rate')
        learning_rate.tag.test_value = 0.0

        #------------------------------------------------------------------
        a_init    = T.zeros((batch_size, n_hid), dtype=np.float32) + c
        post_init = T.zeros(batch_size, dtype=np.float32)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype='float32')
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        return post[-1,:]

    def f_sample(self):
        n_vis, n_hid, batch_size = self.get_hyper_params(['n_vis', 'n_hid', 'batch_size'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])

        a_init    = T.zeros((batch_size, n_hid), dtype=np.float32) + c
        post_init = T.zeros(batch_size, dtype=np.float32)
        vis_init  = T.zeros(batch_size, dtype=np.float32)

        def one_iter(Wi, Vi, bi, a, vis_i, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            vis_i = 1.*(theano_rng.uniform([batch_size]) <= pi)
            post  = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype='float32')
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )

        return vis.T, post[-1,:]

