#!/usr/bin/env python 

from __future__ import division

import sys

import abc
import logging

import numpy as np

import theano 
import theano.tensor as T

import monitor

_logger = logging.getLogger(__name__)

class Termination(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def continue_learning(self, L):
        pass

#-----------------------------------------------------------------------------
class LogLikelihoodIncrease(Termination):
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

#-----------------------------------------------------------------------------
class EarlyStopping(Termination):
    def __init__(self, interval=5, max_epochs=1000):
        super(EarlyStopping, self).__init__()
        
        self.max_epochs = max_epochs
        self.epochs = 0
        self.interval = interval
        self.fails = 0
        self.best_LL = -np.inf

    def reset(self):
        self.epochs = 0
        self.fails = 0
        self.best_LL = -np.inf

    def continue_learning(self, L):
        self.epochs += 1
        L = monitor.validation_LL
        assert isinstance(L, float)

        if self.epochs > self.max_epochs:
            return False


        if L > self.best_LL:
            self.best_LL = L
            self.fails = 0
            increase = (L-self.best_LL)/(np.abs(self.best_LL))
            _logger.info("Validation LL=%5.2f (increased by %f %%)" % (L, 100*increase))
        else:
            self.fails += 1
            _logger.info("Validation LL stagnated (%dth)" % (self.fails))

        return self.fails < self.interval

