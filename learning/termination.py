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
    def __init__(self, min_increase=0.001, interval=5, max_epochs=1000, min_epochs=10):
        super(LogLikelihoodIncrease, self).__init__()
        
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.interval = interval
        self.min_increase = min_increase

        self.reset()
        
    def reset(self):
        self.epochs = 0
        self.L = [-np.inf]*self.interval

    def continue_learning(self, L):
        self.epochs += 1
        self.L.append(L)

        best_in_interval     = np.array(self.L[-self.interval:]).max()
        best_before_interval = np.array(self.L[:-self.interval]).max()
        increase = (best_in_interval-best_before_interval)/(np.abs(best_before_interval))

        if np.isnan(increase):
            increase = +np.inf

        _logger.info("LL increased by %f %%" % (100*increase))
 
        cont = (self.epochs < self.min_epochs) or (increase >= self.min_increase)
        cont = cont and (self.epochs <= self.max_epochs)
        return cont

