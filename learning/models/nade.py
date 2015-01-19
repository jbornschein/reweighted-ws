#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print

from learning.model import default_weights
from learning.models.rws import TopModule, Module, theano_rng
from learning.utils.unrolled_scan import unrolled_scan

_logger = logging.getLogger(__name__)
floatX = theano.config.floatX

#------------------------------------------------------------------------------

class NADETop(TopModule):
    """ Top Level NADE """
    def __init__(self, **hyper_params):
        super(NADETop, self).__init__()

        self.register_hyper_param('n_X', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('unroll_scan', default=1)

        self.register_model_param('b',  help='visible bias', default=lambda: np.zeros(self.n_X))
        self.register_model_param('c',  help='hidden bias' , default=lambda: np.zeros(self.n_hid))
        self.register_model_param('W',  help='encoder weights', default=lambda: default_weights(self.n_X, self.n_hid) )
        self.register_model_param('V',  help='decoder weights', default=lambda: default_weights(self.n_hid, self.n_X) )
        
        self.set_hyper_params(hyper_params)
   
    def setup(self):
        _logger.info("setup")
        if self.n_hid is None:
            self.n_hid = self.n_X

    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor 
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, n_hid = self.get_hyper_params(['n_X', 'n_hid'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])
        
        batch_size = X.shape[0]
        vis = X

        #------------------------------------------------------------------
    
        a_init    = T.zeros([batch_size, n_hid]) + T.shape_padleft(c)
        post_init = T.zeros([batch_size], dtype=floatX)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
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

    def sample(self, n_samples):
        """ Sample from this toplevel module and return X ~ P(X), log(P(X))

        Parameters
        ----------
        n_samples:
            number of samples to drawn

        Returns
        -------
        X:      T.tensor
            samples from this module
        log_p:  T.tensor
            log-probabilities for the samples returned in X
        """
        n_X, n_hid = self.get_hyper_params(['n_X', 'n_hid'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])

        #------------------------------------------------------------------
    
        a_init    = T.zeros([n_samples, n_hid]) + T.shape_padleft(c)
        post_init = T.zeros([n_samples], dtype=floatX)
        vis_init  = T.zeros([n_samples], dtype=floatX)
        rand      = theano_rng.uniform((n_X, n_samples), nstreams=512)

        def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            vis_i = T.cast(rand_i <= pi, floatX)
            post  = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b, rand], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return vis.T, post[-1,:]

#----------------------------------------------------------------------------

class NADE(Module):
    """ Conditional NADE """
    def __init__(self, **hyper_params):
        super(NADE, self).__init__()

        self.register_hyper_param('n_X', help='no. observed binary variables')
        self.register_hyper_param('n_Y', help='no. conditioning binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('unroll_scan', default=1)

        self.register_model_param('b',  help='visible bias', default=lambda: np.zeros(self.n_X))
        self.register_model_param('c',  help='hidden bias' , default=lambda: np.zeros(self.n_hid))
        self.register_model_param('Ub', help='cond. weights Ub', default=lambda: default_weights(self.n_Y, self.n_X) )
        self.register_model_param('Uc', help='cond. weights Uc', default=lambda: default_weights(self.n_Y, self.n_hid) )
        self.register_model_param('W',  help='encoder weights', default=lambda: default_weights(self.n_X, self.n_hid) )
        self.register_model_param('V',  help='decoder weights', default=lambda: default_weights(self.n_hid, self.n_X) )
        
        self.set_hyper_params(hyper_params)

    def setup(self):
        if self.n_hid is None:
            self.n_hid = min(self.n_X, self.n_Y)

    def log_prob(self, X, Y):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer
        X:      T.tensor
            samples from the lower layer

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X and Y
        """
        n_X, n_Y, n_hid    = self.get_hyper_params(['n_X', 'n_Y', 'n_hid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])
        
        batch_size = X.shape[0]
        vis = X
        cond = Y

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
    
        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=floatX)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b_cond.T],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return post[-1,:]
   
    def sample(self, Y):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        X:      T.tensor
            samples from the lower layer        
        log_p:  T.tensor
            log-probabilities for the samples in X and Y
        """
        n_X, n_Y, n_hid = self.get_hyper_params(['n_X', 'n_Y', 'n_hid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])

        batch_size = Y.shape[0]
        cond = Y

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
    
        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=floatX)
        vis_init  = T.zeros([batch_size], dtype=floatX)
        rand      = theano_rng.uniform((n_X, batch_size), nstreams=512)

        def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            vis_i = T.cast(rand_i <= pi, floatX)
            post  = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b_cond.T, rand], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return vis.T, post[-1,:]

