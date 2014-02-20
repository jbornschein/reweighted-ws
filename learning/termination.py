#!/usr/bin/env python 

from __future__ import division

import sys

import logging

import numpy as np

import theano 
import theano.tensor as T

_logger = logging.getLogger(__name__)

class TerminationCriterium(object):
    pass

class LogLikelihoodIncrease(TerminationCriterium):
    def __init__(self, min_increase=0.001, interval=5, max_epochs=1000):
        super(LogLikelihoodIncrease, self).__init__()
        
        self.max_epochs = max_epochs
        self.interval = interval
        self.min_increase = min_increase

        self.reset()
        
    def reset(self):
        self.epochs = 0
        self.L = [-np.inf]

    def continue_learning(self, L):
        self.epochs += 1
        self.L.append(L)

        best_in_interval = np.array(self.L[-self.interval-1:-1]).max()
        increase = best_in_interval/L - 1

        _logger.debug("LL increased by %f %% " % (100*increase))
 
        return (increase >= self.min_increase) and (self.epochs <= self.max_epochs)

