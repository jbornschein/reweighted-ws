#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import Print

from model import Model, default_weights
from utils.unrolled_scan import unrolled_scan

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

theano.config.exception_verbosity = 'high'
theano_rng = RandomStreams(seed=2341)

def gen_binary_matrix(n_bits):
    n_bits = int(n_bits)
    rows = 2**n_bits
    M = np.zeros((rows, n_bits), dtype=floatX)
    for i in xrange(rows):
        for j in xrange(n_bits):
            if i & (1 << j): 
                M[i,7-j] = 1.
    return M

def f_replicate_batch(X, repeat):
    X_ = X.dimshuffle((0, 'x', 1))
    X_ = X_ + T.zeros((X.shape[0], repeat, X.shape[1]), dtype=floatX)
    X_ = X_.reshape( [X_.shape[0]*repeat, X.shape[1]] )
    return X_

class ISB(Model):
    def __init__(self, **hyper_params):
        super(ISB, self).__init__()

        self.register_hyper_param('n_vis', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('n_qhid', help='no. latent binary variables')
        self.register_hyper_param('clamp_sigmoid', default=True)
        self.register_hyper_param('n_samples', default=100)
        self.register_hyper_param('unroll_scan', default=1)

        # Sigmoid Belief Layer
        self.register_model_param('P_a', help='P hidden prior', default=lambda: np.zeros(self.n_hid))
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
        self.Q_params = ['Q_b', 'Q_c', 'Q_Ub', 'Q_Uc', 'Q_W', 'Q_V']
        self.P_params = ['P_a', 'P_b', 'P_W']

    def f_sigmoid(self, x):
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    def f_loglikelihood(self, X, Y=None):
        n_hid, n_samples = self.get_hyper_params(['n_hid', 'n_samples'])
        #b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])

        batch_size = X.shape[0]

        # grow X by factor of samples
        X = f_replicate_batch(X, n_samples)

        # Get samples from q
        #H, lQ = self.f_q_sample_flat(X)   # batch*n_samples  
        H, lQ = self.f_q_sample(X)

        #H  = Print('H')(H)
        #lQ = Print('lQ')(lQ)

        # Calculate log P(X, H)
        lP = self.f_p(X, H)

        def logsumexp1(A):
            A = A.reshape( (batch_size, n_samples) )
            A_max = T.max(A, axis=1)
            A_ = A - T.shape_padright(A_max)
            A = T.log(T.sum(T.exp(A_), axis=1))
            A = A + A_max
            return T.shape_padright(A)

        # Approximate log P(X)
        lPx = logsumexp1(lP-lQ)

        #lP  = Print('lP')(lP)
        #lPx = Print('lPx')(lPx)

        H   =  H.reshape( (batch_size, n_samples, n_hid) )
        lP  = lP.reshape( (batch_size, n_samples) )
        lQ  = lQ.reshape( (batch_size, n_samples) )
        #lQl = lQl.reshape( (batch_size, n_samples) )

        # calc. sampling weights
        w = T.exp(lP-lQ-lPx) 
        #w = Print('w')(w)

        return lP, lQ, H, w

    def f_true_loglikelihood(self, X):
        """ compute the true log-posterior of the datapoints in X
            using a full binary matrix. Will only work for n_hid << 20 
        """
        n_vis, n_hid, n_qhid = self.get_hyper_params(['n_vis', 'n_hid', 'n_qhid'])

        if n_hid > 20:
            _logger.error("You should not call f_true_loglikelihood with n_hid >> 15"
                            " but n_hid=%d" %  n_hid)

        H_all = gen_binary_matrix(n_hid)
        H_all = theano.shared(H_all, name='H_all')

        def one_iter(X_i):
            lP = self.f_p(X_i, H_all)
            lP_max = T.max(lP)
            lP = T.log(T.sum(T.exp(lP-lP_max)))+lP_max
            return lP

        post, updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[X],
                    outputs_info=None, #[a_init],
                    unroll=self.unroll_scan
                )
        return post


    #------------------------ P ---------------------------------------------
    def f_p(self, X, H):
        W, a, b = self.get_model_params(['P_W', 'P_a', 'P_b'])

        # Prior P(H)
        p_hid = self.f_sigmoid(a)
        lpH = T.log(p_hid*H + (1-p_hid)*(1-H))
        lpH = T.sum(lpH, axis=1)
        
        # Posterior P(X|H)
        pX = self.f_sigmoid(T.dot(H, W) + b)
        lpXH = T.log(pX*X + (1-pX)*(1-X))
        lpXH = T.sum(lpXH, axis=1)

        lP = lpH + lpXH
        #lP  = Print('lP')(lP)
        return lP

    def f_p_sample(self, n_samples):
        n_vis, n_hid = self.get_hyper_params(['n_vis', 'n_hid'])
        W, a, b = self.get_model_params(['P_W', 'P_a', 'P_b'])

        # samples hiddens
        p_hid = self.f_sigmoid(a)
        H = T.cast(theano_rng.uniform((n_samples, n_hid)) <= p_hid, dtype=floatX)

        # sample visible given hiddens
        p_vis = self.f_sigmoid(T.dot(H, W) + b)
        X = T.cast(theano_rng.uniform((n_samples, n_vis)) <= p_vis, dtype=floatX)

        return H, X

    #------------------------ Q ---------------------------------------------
    def f_q_sample_flat(self, X):
        n_vis, n_hid, n_qhid = self.get_hyper_params(['n_vis', 'n_hid', 'n_qhid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['Q_b', 'Q_c', 'Q_W', 'Q_V', 'Q_Ub', 'Q_Uc'])

        H_ = gen_binary_matrix(n_hid)
       
        batch_size = 1 # X.shape[0] // 256

        #H_ = f_replicate_batch(theano.shared(H_), batch_size)
        H_ = theano.shared(H_)
        Q_ = theano.shared((np.ones(256*batch_size)/256))
        return H_, Q_

    def f_q_sample(self, X):
        n_vis, n_hid, n_qhid = self.get_hyper_params(['n_vis', 'n_hid', 'n_qhid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['Q_b', 'Q_c', 'Q_W', 'Q_V', 'Q_Ub', 'Q_Uc'])

        cond = X
        batch_size = cond.shape[0]

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)

        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=floatX)
        vis_init  = T.zeros([batch_size], dtype=floatX)
        urand     = theano_rng.uniform([n_hid, batch_size])  # uniform random numbers

        def one_iter(Wi, Vi, bi, urand_i, a, vis_i, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            vis_i = 1.*(urand_i <= pi)
            #post  = T.cast(post + vis_i*T.log(pi) + (1-vis_i)*T.log(1-pi), dtype=floatX)
            post  = post + T.log(vis_i*pi + (1-vis_i)*(1-pi))
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b_cond.T, urand], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )
        return vis.T, post[-1,:]


    def f_q(self, vis, cond):
        n_vis, n_hid, n_qhid = self.get_hyper_params(['n_vis', 'n_hid', 'n_qhid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['Q_b', 'Q_c', 'Q_W', 'Q_V', 'Q_Ub', 'Q_Uc'])

        batch_size = cond.shape[0]

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
    
        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=floatX)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.f_sigmoid(a)
            pi   = self.f_sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype=floatX)
            #post = post + vis_i*T.log(pi) + (1-vis_i)*T.log(1-pi)
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b_cond.T],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        return post[-1,:]

