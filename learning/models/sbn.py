#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print

from learning.models.rws import TopModule, Module, theano_rng
from learning.model import default_weights

_logger = logging.getLogger(__name__)
floatX = theano.config.floatX


class SBNTop(TopModule):
    """ FactoizedBernoulliTop top layer """
    def __init__(self, **hyper_params):
        super(SBNTop, self).__init__()
        
        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')

        # Model parameters
        self.register_model_param('a', help='sigmoid(a) prior', 
            default=lambda: -np.ones(self.n_X))

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
        a, = self.get_model_params(['a'])

        # Calculate log-bernoulli
        prob_X = self.sigmoid(a)
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

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
        a, = self.get_model_params(['a'])

        # sample hiddens
        prob_X = self.sigmoid(a)
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)        
        X = T.cast(U <= prob_X, dtype=floatX)

        return X, self.log_prob(X)

#----------------------------------------------------------------------------

class SBN(Module):
    """ SigmoidBeliefLayer """
    def __init__(self, **hyper_params):
        super(SBN, self).__init__()

        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')

        # Sigmoid Belief Layer
        self.register_model_param('b', help='P lower-layer bias', default=lambda: -np.ones(self.n_X))
        self.register_model_param('W', help='P weights', default=lambda: default_weights(self.n_Y, self.n_X) )

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
        W, b = self.get_model_params(['W', 'b'])

        # posterior P(X|Y)
        prob_X = self.sigmoid(T.dot(Y, W) + b)
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
        W, b = self.get_model_params(['W', 'b'])

        n_samples = Y.shape[0]

        # sample X given Y
        prob_X = self.sigmoid(T.dot(Y, W) + b)
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
        W, b = self.get_model_params(['W', 'b'])

        prob_X = self.sigmoid(T.dot(Y, W) + b)

        return prob_X

