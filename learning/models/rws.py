#!/usr/bin/env python 

from __future__ import division

import logging
from six import iteritems
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

from learning.model import Model
from learning.utils.datalog  import dlog

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

theano.config.exception_verbosity = 'high'
theano_rng = MRG_RandomStreams(seed=2341)


def enumerate_pairs(start, end):
    return [(i, i+1) for i in xrange(0, end-1)]

#=============================================================================

def f_replicate_batch(A, repeat):
    """Extend the given 2d Tensor by repeating reach line *repeat* times.

    With A.shape == (rows, cols), this function will return an array with
    shape (rows*repeat, cols).

    Parameters
    ----------
    A : T.tensor
        Each row of this 2d-Tensor will be replicated *repeat* times
    repeat : int

    Returns
    -------
    B : T.tensor
    """
    A_ = A.dimshuffle((0, 'x', 1))
    A_ = A_ + T.zeros((A.shape[0], repeat, A.shape[1]), dtype=floatX)
    A_ = A_.reshape( [A_.shape[0]*repeat, A.shape[1]] )
    return A_

def f_logsumexp(A, axis=None):
    """Numerically stable log( sum( exp(A) ) ) """
    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(T.sum(T.exp(A-A_max), axis=axis, keepdims=True))+A_max
    B = T.sum(B, axis=axis)
    return B

#=============================================================================

class TopModule(Model):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(TopModule, self).__init__()
        self.register_hyper_param('clamp_sigmoid', default=False)

    def setup(self):
        pass

    def sigmoid(self, x):
        """ Compute the element wise sigmoid function of x 

        Depending on the *clamp_sigmoid* hyperparameter, this might
        return a saturated sigmoid T.nnet.sigmoid(x)*0.9999 + 0.000005
        """
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    @abstractmethod
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
        return X, log_p

    @abstractmethod
    def log_prob(self, X):
        """ Calculate the log-probabilities for the samples in X 

        Parameters
        ----------
        X:      T.tensor
            samples to evaluate

        Returns
        -------
        log_p:  T.tensor
        """
        return log_p

class Module(Model):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Module, self).__init__()

        self.register_hyper_param('clamp_sigmoid', default=False)

    def setup(self):
        pass

    def sigmoid(self, x):
        """ Compute the element wise sigmoid function of x 

        Depending on the *clamp_sigmoid* hyperparameter, this might
        return a saturated sigmoid T.nnet.sigmoid(x)*0.9999 + 0.000005
        """
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    @abstractmethod
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
        X, log_p = None, None
        return X, log_p

    @abstractmethod
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
        return log_p

#=============================================================================

class LayerStack(Model):
    def __init__(self, **hyper_params):     
        super(LayerStack, self).__init__()

        # Hyper parameters
        self.register_hyper_param('p_layers', help='STBP P layers', default=[])
        self.register_hyper_param('q_layers', help='STBP Q layers', default=[])
        self.register_hyper_param('n_samples', help='no. of samples to use', default=10)

        self.set_hyper_params(hyper_params)

    def setup(self):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)
        
        assert len(p_layers) == len(q_layers)+1
        assert isinstance(p_layers[-1], TopModule)

        
        self.n_X = p_layers[0].n_X

        for l in xrange(0, n_layers-1):
            assert isinstance(p_layers[l], Module)
            assert isinstance(q_layers[l], Module)
            assert p_layers[l].n_Y == p_layers[l+1].n_X
            assert p_layers[l].n_Y == q_layers[l].n_X 

            p_layers[l].setup()
            q_layers[l].setup()


        p_layers[-1].setup()

    def sample_p(self, n_samples):
        """ Draw *n_samples* drawn from the P-model.
        
        This method returns a list with the samples values on all layers
        and the correesponding log_p.
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        # Generate samples from the generative model
        samples = [None]*n_layers

        samples[-1], log_prob = p_layers[-1].sample(n_samples)
        for l in xrange(n_layers-1, 0, -1):
            samples[l-1], log_p_l = p_layers[l-1].sample(samples[l])
            log_prob += log_p_l
        
        return samples, log_prob

    def sample_q(self, X, Y=None):
        """ Given a set of observed X, samples from q(H | X) and calculate 
            both P(X, H) and Q(H | X)
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        size = X.shape[0]

        # Prepare input for layers
        samples = [None]*n_layers
        log_q   = [None]*n_layers
        log_p   = [None]*n_layers

        samples[0] = X
        log_q[0]   = T.zeros([size]) 

        # Generate samples (feed-forward)
        for l in xrange(n_layers-1):
            samples[l+1], log_q[l+1] = q_layers[l].sample(samples[l])
        
        # Get log_probs from generative model
        log_p[n_layers-1] = p_layers[n_layers-1].log_prob(samples[n_layers-1])
        for l in xrange(n_layers-1, 0, -1):
            log_p[l-1] = p_layers[l-1].log_prob(samples[l-1], samples[l])

        return samples, log_p, log_q
 
    def log_likelihood(self, X, Y=None, n_samples=None):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        if n_samples == None:
            n_samples = self.n_samples

        batch_size = X.shape[0]

        # Get samples
        X = f_replicate_batch(X, n_samples)
        samples, log_p, log_q = self.sample_q(X, None)

        # Reshape and sum
        log_p_all = T.zeros((batch_size, n_samples))
        log_q_all = T.zeros((batch_size, n_samples))
        for l in xrange(n_layers):
            samples[l] = samples[l].reshape((batch_size, n_samples, p_layers[l].n_X))
            log_q[l] = log_q[l].reshape((batch_size, n_samples))
            log_p[l] = log_p[l].reshape((batch_size, n_samples))
            log_p_all += log_p[l]   # agregate all layers
            log_q_all += log_q[l]   # agregate all layers

        # Approximate log P(X)
        log_px = f_logsumexp(log_p_all-log_q_all, axis=1) - T.log(n_samples)
        
        # Calculate samplig weights
        log_pq = (log_p_all-log_q_all-T.log(n_samples))
        w_norm = f_logsumexp(log_pq, axis=1)
        log_w = log_pq-T.shape_padright(w_norm)
        w = T.exp(log_w)

        # Calculate KL(P|Q), Hp, Hq
        KL = [None]*n_layers
        Hp = [None]*n_layers
        Hq = [None]*n_layers
        for l in xrange(n_layers):
            KL[l] = T.sum(w*(log_p[l]-log_q[l]), axis=1)
            Hp[l] = f_logsumexp(log_w+log_p[l], axis=1)
            Hq[l] = T.sum(w*log_q[l], axis=1)

        return log_px, w, log_p_all, log_q_all, KL, Hp, Hq

    def get_gradients(self, X, Y, lr_p, lr_q, n_samples):
        """ return log_PX and an OrderedDict with parameter gradients """
        log_PX, w, log_p, log_q, KL, Hp, Hq = self.log_likelihood(X, Y, n_samples=n_samples)
        
        batch_log_PX = T.sum(log_PX)
        cost_p = T.sum(T.sum(log_p*w, axis=1))
        cost_q = T.sum(T.sum(log_q*w, axis=1))

        gradients = OrderedDict()
        for nl, layer in enumerate(self.p_layers):
            for name, shvar in iteritems(layer.get_model_params()):
                gradients[shvar] = lr_p[nl] * T.grad(cost_p, shvar, consider_constant=[w])

        for nl, layer in enumerate(self.q_layers):
            for name, shvar in iteritems(layer.get_model_params()):
                gradients[shvar] = lr_q[nl] * T.grad(cost_q, shvar, consider_constant=[w])

        return batch_log_PX, gradients

    def get_sleep_gradients(self, lr_s=1., n_dreams=100):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        p, log_p = self.sample_p(n_dreams)

        log_q = T.zeros((n_dreams,))
        for i, j in enumerate_pairs(0, n_layers):
            log_q += q_layers[i].log_prob(p[i+1], p[i])

        cost_q = T.sum(log_q)

        gradients = OrderedDict()
        for nl, layer in enumerate(self.q_layers):
            for name, shvar in iteritems(layer.get_model_params()):
                gradients[shvar] = lr_s[nl] * T.grad(cost_q, shvar)

        return log_q, gradients
        
    #------------------------------------------------------------------------
    def get_p_params(self):
        params = OrderedDict()
        for l in self.p_layers:
            params.update( l.get_model_params() )
        return params

    def get_q_params(self):
        params = OrderedDict()
        for l in self.q_layers:
            params.update( l.get_model_params() )
        return params

    def model_params_to_dict(self):
        vals = {}
        for n,l in enumerate(self.p_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.P.%s" % (n, pname)
                vals[key] = shvar.get_value()
        for n,l in enumerate(self.q_layers):                
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.Q.%s" % (n, pname)
                vals[key] = shvar.get_value()
        return vals

    def model_params_from_dict(self, vals):
        for n,l in enumerate(self.p_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.P.%s" % (n, pname)
                value = vals[key]
                shvar.set_value(value)
        for n,l in enumerate(self.q_layers):                
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.Q.%s" % (n, pname)
                value = vals[key]
                shvar.set_value(value)

    def model_params_to_dlog(self, dlog):
        vals = self.model_params_to_dict()
        dlog.append_all(vals)

    def model_params_from_dlog(self, dlog, row=-1):
        for n,l in enumerate(self.p_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.P.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)
        for n,l in enumerate(self.q_layers):                
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.Q.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)
 
    def model_params_from_h5(self, h5, row=-1, basekey="model."):
        for n,l in enumerate(self.p_layers):
            try:
                for pname, shvar in iteritems(l.get_model_params()):
                    key = "%sL%d.P.%s" % (basekey, n, pname)
                    value = h5[key][row]
                    shvar.set_value(value)
            except KeyError:
                if n >= len(self.p_layers)-2:
                    _logger.warning("Unable to load top P-layer params %s[%d]... continuing" % (key, row))
                    continue
                else:
                    _logger.error("Unable to load %s[%d] from %s" % (key, row, h5.filename))
                    raise

        for n,l in enumerate(self.q_layers):                
            try:
                for pname, shvar in iteritems(l.get_model_params()):
                    key = "%sL%d.Q.%s" % (basekey, n, pname)
                    value = h5[key][row]
                    shvar.set_value(value)
            except KeyError:
                if n == len(self.q_layers)-1:
                    _logger.warning("Unable to load top Q-layer params %s[%d]... continuing" % (key, row))
                    continue
                _logger.error("Unable to load %s[%d] from %s" % (key, row, h5.filename))
                raise
                    

