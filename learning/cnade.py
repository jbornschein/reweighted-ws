#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import Model, default_weights
from utils.unrolled_scan import unrolled_scan

_logger = logging.getLogger(__name__)



theano.config.exception_verbosity = 'high'
theano_rng = RandomStreams(seed=2341)

class CNADE(Model):
    def __init__(self, **hyper_params):
        super(CNADE, self).__init__()

        self.register_hyper_param('n_vis', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('n_cond', help='no. conditioning binary variables')
        self.register_hyper_param('clamp_sigmoid', default=False)
        self.register_hyper_param('unroll_scan', default=1)

        self.register_model_param('b',  help='visible bias', default=lambda: np.zeros(self.n_vis))
        self.register_model_param('c',  help='hidden bias' , default=lambda: np.zeros(self.n_hid))
        self.register_model_param('Ub', help='cond. weights Ub', default=lambda: default_weights(self.n_cond, self.n_vis) )
        self.register_model_param('Uc', help='cond. weights Uc', default=lambda: default_weights(self.n_cond, self.n_hid) )
        self.register_model_param('W',  help='encoder weights', default=lambda: default_weights(self.n_vis, self.n_hid) )
        self.register_model_param('V',  help='decoder weights', default=lambda: default_weights(self.n_hid, self.n_vis) )
        
        self.set_hyper_params(hyper_params)

    def f_sigmoid(self, x):
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    def f_loglikelihood(self, X, Y):
        n_vis, n_hid, n_cond = self.get_hyper_params(['n_vis', 'n_hid', 'n_cond'])
        b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])
        
        batch_size = X.shape[0]
        vis = X
        cond = Y

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
    
        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=np.float32)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype='float32')
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b_cond.T],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        return post[-1,:]

    def f_sample(self, Y):
        n_vis, n_hid, n_cond = self.get_hyper_params(['n_vis', 'n_hid', 'n_cond'])
        b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])

        batch_size = Y.shape[0]
        cond = Y

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
    
        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=np.float32)
        vis_init  = T.zeros([batch_size], dtype=np.float32)

        def one_iter(Wi, Vi, bi, a, vis_i, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            vis_i = 1.*(theano_rng.uniform([batch_size]) <= pi)
            post  = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype='float32')
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b_cond.T], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )

        return vis.T, post[-1,:]

