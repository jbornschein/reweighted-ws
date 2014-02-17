#!/usr/bin/env python 

from __future__ import division

import logging
from collections import namedtuple
from recordtype import recordtype

import numpy as np

import theano 
import theano.tensor as T

_logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------

def default_weights():
    pass

#------------------------------------------------------------------------------

HyperParam = recordtype('HyperParam', 'value name default help')
ModelParam = recordtype('ModelParam', 'value name help')

#------------------------------------------------------------------------------

class Model(object):
    _model_params = None
    _hyper_params = None

    #--------------------------------------------------------------------------

    def __init__(self, hyper_params={}):
        self.set_hyper_param(hyper_params)

    def _ensure_init(self):
        if self._hyper_params is None:
            self._hyper_params = {}
        if self._model_params is None:
            self._model_params = {}

    def register_hyper_param(self, key, default=None, help=None):
        self._ensure_init()
        if self._hyper_params.has_key(key):
            raise ValueError('A hyper parameter named "%s" already exists' % key)
        if self._model_params.has_key(key):
            raise ValueError('A model parameter named "%s" already exists' % key)

        self._hyper_params[key] = HyperParam(name=key, value=default, default=default, help=help)
        
    def register_model_param(self, key, help=None):
        self._ensure_init()
        if self._model_params is None:
            self._model_params = {}
        if self._hyper_params.has_key(key):
            raise ValueError('A hyper parameter named "%s" already exists' % key)
        if self._model_params.has_key(key):
            raise ValueError('A model parameter named "%s" already exists' % key)

        self._model_params[key] = ModelParam(name=key, value=None, help=help)

    #--------------------------------------------------------------------------

    def get_hyper_param(self, key):
        """ Return the value of a predefined hyper parameter.

        `key` may be  list or a tuple in which case this method
        will return a list with the specified parameter values.
        """
        if isinstance(key, (list, tuple)):  
            return [self.get_hyper_param(k) for k in key]

        param = self._hyper_params.get(key, None)
        if param is None:
            raise ValueError('Trying to access unknown hyper parameter "%s"' % key)
        return param.value

    def get_model_param(self, key):
        """ Return the value of a predefined model parameter.

        `key` may be  list or a tuple in which case this method
        will return a list with the specified parameter values.
        """
        if isinstance(key, (list, tuple)):  
            return [self.get_model_param(k) for k in key]

        param = self._model_params.get(key, None)
        if param is None:
            raise ValueError('Trying to access unknown model parameter "%s"' % key)
        #assert isinstance(param.vale, theano.Shared)
        return param.value

    def set_hyper_param(self, key, val=None):
        if isinstance(key, dict):
            for key, val in key.iteritems():
                self.set_hyper_param(key, val)
        
        param = self._hyper_params.get(key, None)
        if param is None:
            raise ValueError('Trying to set unknown hyper parameter "%s"' % key)
        param.value = val
        
    def set_model_param(self, key, val=None):
        if isinstance(key, dict):
            for key, val in key.iteritems():
                self.set_model_param(key, val)
        
        param = self._model_params.get(key, None)
        if param is None:
            raise ValueError('Trying to set unknown model parameter "%s"' % key)
        param.value = val
 
#------------------------------------------------------------------------------

