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

def f_replicate_batch(X, repeat):
    X_ = X.dimshuffle((0, 'x', 1))
    X_ = X_ + T.zeros((X.shape[0], repeat, X.shape[1]), dtype='float32')
    X_ = X_.reshape( [X_.shape[0]*repeat, X.shape[1]] )
    return X_


class ISB(Model):
    def __init__(self, **hyper_params):
        super(ISB, self).__init__()

        self.register_hyper_param('n_vis', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('n_qhid', help='no. latent binary variables')
        self.register_hyper_param('clamp_sigmoid', default=False)
        self.register_hyper_param('n_samples', default=100)
        self.register_hyper_param('unroll_scan', default=1)

        # Sigmoid Belief Layer
        self.register_model_param('P_a', help='P hidden prior', default=lambda: np.ones(self.n_hid)/2.)
        self.register_model_param('P_b', help='P visible bias', default=lambda: np.zeros(self.n_vis))
        self.register_model_param('P_W', help='P weights', default=lambda: default_weights(self.n_hid, self.n_vis) )

        # Conditional NADE
        self.register_model_param('Q_b',  help='Q visible bias', default=lambda: np.zeros(self.n_hid))
        self.register_model_param('Q_c',  help='Q hidden bias' , default=lambda: np.zeros(self.n_qhid))
        self.register_model_param('Q_Ub', help='Q cond. weights Ub', default=lambda: default_weights(self.n_vis, self.n_hid) )
        self.register_model_param('Q_Uc', help='Q cond. weights Uc', default=lambda: default_weights(self.n_vis, self.n_qhid) )
        self.register_model_param('Q_W',  help='Q encoder weights', default=lambda: default_weights(self.n_hid, self.n_qhid) )
        self.register_model_param('Q_V',  help='Q decoder weights', default=lambda: default_weights(self.n_qhid, self.n_hid) )

        self.set_hyper_params(hyper_params)

    def f_sigmoid(self, x):
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    def f_q(self, X, Y):
        n_vis, n_qhid, n_cond = self.get_hyper_params(['n_vis', 'n_qhid', 'n_cond'])
        b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])
        
        batch_size = X.shape[0]

        vis = X
        cond = Y

        learning_rate = T.fscalar('learning_rate')
        learning_rate.tag.test_value = 0.0

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

    def f_loglikelihood(self, X, Y=None):
        n_samples, = self.get_hyper_params(['n_samples'])
        #b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])

        batch_size = X.shape[0]

        # grow X by factor of samples
        X = f_replicate_batch(X, n_samples)

        # Get samples from q
        H, lQ = self.f_q_sample(X)   # batch*n_samples  

        # Calculate log P(X, H)
        lP = self.f_p(X, H)

        # Approximate log P(X)
        lPx = T.shape_padright(lP)
        lPx = lPx.reshape( (batch_size, n_samples) )
        lPx_max = T.max(lPx, axis=1)
        lPx = lPx - T.shape_padright(lPx_max)
        lPx = T.log(T.sum(T.exp(lPx), axis=1))
        lPx = T.shape_padright(lPx)
        #lPx = f_replicate_batch(lPx, n_samples)

        lP = lP.reshape( (batch_size, n_samples) )
        lQ = lQ.reshape( (batch_size, n_samples) )

        # calc. sampling weights
        w = T.exp(lP-lQ-lPx) 

        print "lP:  ", lP.tag.test_value.shape
        print "lQ:  ", lQ.tag.test_value.shape
        print "lPx: ", lPx.tag.test_value.shape
        print "w:   ", w.tag.test_value.shape


        return lP, lQ, H, w

    #------------------------ P ---------------------------------------------
    def f_p(self, X, H):
        W, a, b = self.get_model_params(['P_W', 'P_a', 'P_b'])

        # Prior P(H)
        lpH = T.cast(T.log(a*H + (1-a)*(1-H)), dtype='float32')
        lpH = T.sum(lpH, axis=1)
        
        # Posterior P(X|H)
        pX = self.f_sigmoid(T.dot(H, W) + b)
        lpXH = T.cast(T.log(pX*X + (1-pX)*(1-X)), dtype='float32')
        lpXH = T.sum(lpXH, axis=1)

        return lpH + lpXH

    #------------------------ Q ---------------------------------------------
    def f_q_sample(self, X):
        n_vis, n_hid, n_qhid = self.get_hyper_params(['n_vis', 'n_hid', 'n_qhid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['Q_b', 'Q_c', 'Q_W', 'Q_V', 'Q_Ub', 'Q_Uc'])

        cond = X
        batch_size = cond.shape[0]

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

