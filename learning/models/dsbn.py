#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print

from learning.model import default_weights
from learning.models.rws import TopModule, Module, theano_rng

_logger = logging.getLogger(__name__)
floatX = theano.config.floatX


class DSBN(Module):
    """ SigmoidBeliefNet with a deterministic layers """
    def __init__(self, **hyper_params):
        super(DSBN, self).__init__()

        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')

        self.register_hyper_param('n_D', help='no. deterministic units')
        self.register_hyper_param('non_lin', default='sigmoid', help="nonlinearity for deterministic layer")

        # Sigmoid Belief Layer
        self.register_model_param('a', help='deterministic bias', default=lambda: -np.ones(self.n_D))
        self.register_model_param('U', help='deterministic weights', default=lambda: default_weights(self.n_Y, self.n_D) )

        self.register_model_param('b', help='stochastic bias', default=lambda: -np.ones(self.n_X))
        self.register_model_param('W', help='stochastic weights', default=lambda: default_weights(self.n_D, self.n_X) )

        self.set_hyper_params(hyper_params)


    def non_linearity(self, arr):
        """ Elementwise non linerity according to self.non_lin """
        non_lin = self.get_hyper_param('non_lin')

        if non_lin == 'tanh':
            return T.tanh(arr)
        elif self.non_lin == 'sigmoid':
            return self.sigmoid(arr)
        else:
            raise ValueError("Unknown non_lin")

    def setup(self):
        if self.n_D is None:
            self.n_D = self.n_Y + self.n_X / 2

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
        U, a = self.get_model_params(['U', 'a'])
        W, b = self.get_model_params(['W', 'b'])

        # posterior P(X|Y)
        D = self.non_linearity(T.dot(Y, U) + a)

        prob_X = self.sigmoid(T.dot(D, W) + b)
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = T.sum(log_prob, axis=1)

        return log_prob

    def sample(self, Y):
        """ Given samples from the upper layer Y, sample values from X
            and return then together with their log probability.

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        X:      T.tensor
            samples from the lower layer
        log_p:  T.tensor
            log-posterior for the samples returned in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        U, a = self.get_model_params(['U', 'a'])
        W, b = self.get_model_params(['W', 'b'])

        n_samples = Y.shape[0]

        # sample X given Y
        D = self.non_linearity(T.dot(Y, U) + a)

        prob_X = self.sigmoid(T.dot(D, W) + b)
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)
        X = T.cast(U <= prob_X, dtype=floatX)

        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

        return X, log_prob

    def sample_expected(self, Y):
        """ Given samples from the upper layer Y, return 
            the probability for the individual X elements

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        X:      T.tensor
        """
        U, a = self.get_model_params(['U', 'a'])
        W, b = self.get_model_params(['W', 'b'])

        D = self.non_linearity(T.dot(Y, U) + a)
        prob_X = self.sigmoid(T.dot(D, W) + b)

        return prob_X

