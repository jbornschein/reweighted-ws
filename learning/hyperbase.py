#!/usr/bin/env python 

from __future__ import division

import logging
from inspect import isfunction
from collections import OrderedDict
from recordtype import recordtype
from six import iteritems

import numpy as np

import theano 
import theano.tensor as T

_logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------

HyperParam = recordtype('HyperParam', 'value name default help')

#------------------------------------------------------------------------------

class HyperBase(object):
    initialized = False

    def __init__(self, **hyper_params):
        self._hyper_params = OrderedDict()
        self.initialized = True

    def _ensure_init(self):
        if not self.initialized:
            raise ArgumentError("HyperBase base class not initialized yet!"
                    "Call yperBase.__init__()  before doing anything else!")

    def register_hyper_param(self, key, default=None, help=None):
        self._ensure_init()
        if self._hyper_params.has_key(key):
            raise ValueError('A hyper parameter named "%s" already exists' % key)

        self._hyper_params[key] = HyperParam(name=key, value=None, default=default, help=help)
        
    #--------------------------------------------------------------------------

    def get_hyper_param(self, key):
        """ Return the value of a predefined hyper parameter. """
        param = self._hyper_params.get(key, None)
        if param is None:
            raise ValueError('Trying to access unknown hyper parameter "%s"' % key)
        if param.value is None:
            if isfunction(param.default):
                self.set_hyper_param(key, param.default())
            else:
                self.set_hyper_param(key, param.default)
        return param.value

    def get_hyper_params(self, keys=None):
        """ """
        if keys is None:
            keys = self._hyper_params.keys()
            return {k: self.get_hyper_param(k) for k in keys}
        else:
            return [self.get_hyper_param(k) for k in keys]

    def set_hyper_param(self, key, val=None):
        param = self._hyper_params.get(key, None)
        if param is None:
            raise ValueError('Trying to set unknown hyper parameter "%s"' % key)
        param.value = val

    def set_hyper_params(self, d):
        for key, val in iteritems(d):
            self.set_hyper_param(key, val)

    #------------------------------------------------------------------------

    def __getattr__(self, name):
        if not self.initialized:
            raise AttributeError("'%s' object has no attribute '%s'" % (repr(self), name))
    
        if name in self._hyper_params:
             return self.get_hyper_param(name)
        raise AttributeError("'%s' object has no attribute '%s'" % (repr(self), name))
    
    def __setattr__(self, name, value):
        if not self.initialized:
            return object.__setattr__(self, name, value)
   
        if name in self._hyper_params:
            return self.set_hyper_param(name, value)
        return object.__setattr__(self, name, value)

#------------------------------------------------------------------------------

