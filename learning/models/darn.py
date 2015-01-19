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

class DARNTop(TopModule):
    def __init__(self, **hyper_params):
        super(DARNTop, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')
        self.register_hyper_param('unroll_scan', default=1)        

        # Model parameters
        self.register_model_param('b', help='sigmoid(b)-bias ', default=lambda: np.zeros(self.n_X))
        self.register_model_param('W', help='weights (triangular)', default=lambda: 0.5*default_weights(self.n_X, self.n_X) )

        self.set_hyper_params(hyper_params)

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
        n_X, = self.get_hyper_params(['n_X'])
        b, W = self.get_model_params(['b', 'W'])
        
        W = T.tril(W, k=-1)

        prob_X = self.sigmoid(T.dot(X, W) + b)
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = T.sum(log_prob, axis=1)

        return log_prob


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
        n_X, = self.get_hyper_params(['n_X'])
        b, W = self.get_model_params(['b', 'W'])

        #------------------------------------------------------------------

        a_init    = T.zeros([n_samples, n_X]) + T.shape_padleft(b)
        post_init = T.zeros([n_samples], dtype=floatX)
        x_init    = T.zeros([n_samples], dtype=floatX)
        rand      = theano_rng.uniform((n_X, n_samples), nstreams=512)

        def one_iter(i, Wi, rand_i, a, X, post):
            pi   = self.sigmoid(a[:,i])
            xi   = T.cast(rand_i <= pi, floatX)
            post = post + T.log(pi*xi + (1-pi)*(1-xi))            
            a    = a + T.outer(xi, Wi) 
            return a, xi, post

        [a, X, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[T.arange(n_X), W, rand],
                    outputs_info=[a_init, x_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return X.T, post[-1,:]


class DARN(Module):
    def __init__(self, **hyper_params):
        super(DARN, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')
        self.register_hyper_param('n_Y', help='no. conditioning binary variables')        
        self.register_hyper_param('unroll_scan', default=1)        

        # Model parameters
        self.register_model_param('b', help='sigmoid(b)-bias ', default=lambda: np.zeros(self.n_X))
        self.register_model_param('W', help='weights (triangular)', default=lambda: default_weights(self.n_X, self.n_X) )
        self.register_model_param('U', help='cond. weights U', default=lambda: default_weights(self.n_Y, self.n_X) )

        self.set_hyper_params(hyper_params)


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
        n_X, n_Y = self.get_hyper_params(['n_X', 'n_Y'])
        b, W, U  = self.get_model_params(['b', 'W', 'U'])
        
        W = T.tril(W, k=-1)

        prob_X = self.sigmoid(T.dot(X, W) + T.dot(Y, U) + T.shape_padleft(b))
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = T.sum(log_prob, axis=1)

        return log_prob


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
        n_X, n_Y = self.get_hyper_params(['n_X', 'n_Y'])
        b, W, U = self.get_model_params(['b', 'W', 'U'])

        batch_size = Y.shape[0]

        #------------------------------------------------------------------

        a_init    = T.dot(Y, U) + T.shape_padleft(b)   # shape (batch, n_vis)
        post_init = T.zeros([batch_size], dtype=floatX)
        x_init    = T.zeros([batch_size], dtype=floatX)
        rand      = theano_rng.uniform((n_X, batch_size), nstreams=512)

        def one_iter(i, Wi, rand_i, a, X, post):
            pi   = self.sigmoid(a[:,i])
            xi   = T.cast(rand_i <= pi, floatX)
            post = post + T.log(pi*xi + (1-pi)*(1-xi))            
            a    = a + T.outer(xi, Wi) 
            return a, xi, post

        [a, X, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[T.arange(n_X), W, rand],
                    outputs_info=[a_init, x_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return X.T, post[-1,:]


