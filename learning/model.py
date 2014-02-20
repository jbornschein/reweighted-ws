#!/usr/bin/env python 

from __future__ import division

import logging
from inspect import isfunction
from collections import OrderedDict
from recordtype import recordtype

import numpy as np

import theano 
import theano.tensor as T

_logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------

def default_weights(n_in, n_out):
    return (2*np.random.normal( size=(n_in, n_out))-1) / n_in

#------------------------------------------------------------------------------

HyperParam = recordtype('HyperParam', 'value name default help')
ModelParam = recordtype('ModelParam', 'value name default help')

#------------------------------------------------------------------------------

class Model(object):
    initialized = False

    def __init__(self, **hyper_params):
        self._model_params = OrderedDict()
        self._hyper_params = OrderedDict()
        self.initialized = True

    def _ensure_init(self):
        if not self.initialized:
            raise ArgumentError("Model base class not initialized yet!"
                    "Call Model.__init__()  before doing anything else!")

    def register_hyper_param(self, key, default=None, help=None):
        self._ensure_init()
        if self._hyper_params.has_key(key):
            raise ValueError('A hyper parameter named "%s" already exists' % key)
        if self._model_params.has_key(key):
            raise ValueError('A model parameter named "%s" already exists' % key)

        self._hyper_params[key] = HyperParam(name=key, value=None, default=default, help=help)
        
    def register_model_param(self, key, default=None, help=None):
        self._ensure_init()
        if self._hyper_params.has_key(key):
            raise ValueError('A hyper parameter named "%s" already exists' % key)
        if self._model_params.has_key(key):
            raise ValueError('A model parameter named "%s" already exists' % key)

        self._model_params[key] = ModelParam(name=key, value=None, default=default, help=help)

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
        return [self.get_hyper_param(k) for k in keys]

    def set_hyper_param(self, key, val=None):
        param = self._hyper_params.get(key, None)
        if param is None:
            raise ValueError('Trying to set unknown hyper parameter "%s"' % key)
        param.value = val

    def set_hyper_params(self, d):
        for key, val in d.iteritems():
            self.set_hyper_param(key, val)

    #------------------------------------------------------------------------

    def get_model_param(self, key):
        """ Return the value of a predefined model parameter. """
        param = self._model_params.get(key, None)
        if param is None:
            raise ValueError('Trying to access unknown model parameter "%s"' % key)
        if param.value is None:
            if isfunction(param.default):
                self.set_model_param(key, param.default())
            else:
                self.set_model_param(key, param.default)
        return param.value

    def get_model_params(self, keys=None):
        """ """
        if keys is None:
            return OrderedDict( [(key, self.get_model_param(key)) for key in self._model_params.keys()] )
        else:
            return [self.get_model_param(k) for k in keys]
 
    def set_model_param(self, key, val=None):
        param = self._model_params.get(key, None)
        if param is None:
            raise ValueError('Trying to set unknown model parameter "%s"' % key)
        if not isinstance(val, T.sharedvar.SharedVariable):
            val = np.asarray(val, dtype='float32')
            val = theano.shared(val, key)
            val.tag.test_value = val
        param.value = val
     
    def set_model_params(self, d):
         for key, val in d.iteritems():
            self.set_model_param(key, val)

    #------------------------------------------------------------------------

    def __getattr__(self, name):
        if not self.initialized:
            raise AttributeError("'%s' object has no attribute '%s'" % (repr(self), name))
    
        if name in self._model_params:
             return self.get_model_param(name)
        if name in self._hyper_params:
             return self.get_hyper_param(name)
        raise AttributeError("'%s' object has no attribute '%s'" % (repr(self), name))
    
    def __setattr__(self, name, value):
        if not self.initialized:
            return object.__setattr__(self, name, value)
   
        if name in self._model_params:
            return self.set_model_param(name, value)
        if name in self._hyper_params:
            return self.set_hyper_param(name, value)
        return object.__setattr__(self, name, value)

#------------------------------------------------------------------------------

