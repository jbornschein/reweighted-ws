#!/usr/bin/env python 

from __future__ import division

import logging
from six import iteritems
from inspect import isfunction
from collections import OrderedDict
from recordtype import recordtype

import numpy as np

import theano 
import theano.tensor as T

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

#------------------------------------------------------------------------------

def default_weights(n_in, n_out):
    """ Return a n_in * n_out shaped matrix with uniformly sampled elements 
        between - and + sqrt(6)/sqrt(n_in+n_out).
    """
    scale = np.sqrt(6) / np.sqrt(n_in+n_out)
    return scale*(2*np.random.uniform(size=(n_in, n_out))-1) / n_in

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
            raise ValueError("Model base class not initialized yet!"
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
            keys = self._model_params.keys()
            return OrderedDict( [(k, self.get_model_param(k)) for k in keys] )
        else:
            return [self.get_model_param(k) for k in keys]
 
    def set_model_param(self, key, val=None):
        param = self._model_params.get(key, None)
        if param is None:
            raise ValueError('Trying to set unknown model parameter "%s"' % key)
        if not isinstance(val, T.sharedvar.SharedVariable):
            if not isinstance(val, np.ndarray):
                val = np.asarray(val)
            if val.dtype == np.float:
                val = np.asarray(val, dtype=floatX)
            val = theano.shared(val, key)
            val.tag.test_value = val
        param.value = val
     
    def set_model_params(self, d):
         for key, val in iteritems(d):
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

    #------------------------------------------------------------------------
    
    def model_params_from_dlog(self, dlog, row=-1):
        """ Load the model params form an open H5 file """
        for key, param in iteritems(self._model_params):
            assert isinstance(param, ModelParam)
            value = dlog.load(key, row)
            shvar = para.value
            shvar.set_value(value)

    def model_params_to_dlog(self, dlog):
        """ Append all model params to dlog """
        for key, param in iteritems(self._model_params):
            assert isinstance(param, HyperParam)
            shvar = param.value
            value = shvar.get_value()
            dlog.append(key, value)

    def hyper_params_from_dlog(self, dlog, row=-1):
        """ Load the hyper params form an open H5 file """
        for key, param in iteritems(self._hyper_params):
            assert isinstance(param, HyperParam)
            value = dlog.load(key, row)
            self.set_hyper_param(key, value)

    def hyper_params_to_dlog(self, dlog):
        """ Append all hyper params to dlog """
        for key, param in iteritems(self._hyper_params):
            assert isinstance(param, ModelParam)
            shvar = param.value
            value = shvar.get_value()
            dlog.append(key, value)

#------------------------------------------------------------------------------

