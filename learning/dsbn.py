#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print

from model import Model, default_weights
from rws import TopModule, Module, theano_rng
from utils.unrolled_scan import unrolled_scan

_logger = logging.getLogger(__name__)
floatX = theano.config.floatX


class DSBN(Module):
    """ SigmoidBeliefLayer """
    def __init__(self, **hyper_params):
        super(DSBN, self).__init__()

        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')
        self.register_hyper_param('n_hid', default=0, help='no. of deterministic hidden units')
        self.register_hyper_param('nonlin', default='tanh', help='non-linearity')

        # Sigmoid Belief Layer
        self.register_model_param('Wd', help='', default=lambda: default_weights(self.n_Y, self.n_hid) )
        self.register_model_param('bd', help='', default=lambda: -np.ones(self.n_hid))
        self.register_model_param('W',  help='P weights', default=lambda: default_weights(self.n_hid, self.n_X) )
        self.register_model_param('b',  help='P lower-layer bias', default=lambda: -np.ones(self.n_X))

        self.set_hyper_params(hyper_params)

        if self.n_hid == 0:
            self.n_hid = 2*min(self.n_X, self.n_Y)

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
        W,  b  = self.get_model_params(['W', 'b'])
        Wd, bd = self.get_model_params(['Wd', 'bd'])


        # posterior P(X|Y)
        Y_hid = T.tanh(T.dot(Y, Wd) + bd)
        #Y_hid = self.sigmoid(T.dot(Y, Wd) + bd)
        prob_X = self.sigmoid(T.dot(Y_hid, W) + b)
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
        n_X,   = self.get_hyper_params(['n_X'])
        W,  b  = self.get_model_params(['W', 'b'])
        Wd, bd = self.get_model_params(['Wd', 'bd'])

        n_samples = Y.shape[0]

        # sample X given Y
        Y_hid = T.tanh(T.dot(Y, Wd) + bd)
        #Y_hid = self.sigmoid(T.dot(Y, Wd) + bd)
        prob_X = self.sigmoid(T.dot(Y_hid, W) + b)
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)
        X = T.cast(U <= prob_X, dtype=floatX)

        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

        return X, log_prob

    def prob_sample(self, Y):
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
        W,  b  = self.get_model_params(['W', 'b'])
        Wd, bd = self.get_model_params(['Wd', 'bd'])

        Y_hid = T.tanh(T.dot(Y, Wd) + bd)
        #Y_hid = self.sigmoid(T.dot(Y, Wd) + bd)
        prob_X = self.sigmoid(T.dot(Y_hid, W) + b)

        return prob_X

