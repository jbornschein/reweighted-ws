#!/usr/bin/env python 

from __future__ import division

import abc
import logging
from collections import OrderedDict

import numpy as np

import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import Print

from model import Model, default_weights
from cnade import CNADE
from utils.unrolled_scan import unrolled_scan
from utils.datalog  import dlog

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

theano.config.exception_verbosity = 'high'
theano_rng = RandomStreams(seed=2341)

#def gen_binary_matrix(n_bits):
#    n_bits = int(n_bits)
#    rows = 2**n_bits
#    M = np.zeros((rows, n_bits), dtype=floatX)
#    for i in xrange(rows):
#        for j in xrange(n_bits):
#            if i & (1 << j): 
#                M[i,7-j] = 1.
#    return M

def f_replicate_batch(X, repeat):
    X_ = X.dimshuffle((0, 'x', 1))
    X_ = X_ + T.zeros((X.shape[0], repeat, X.shape[1]), dtype=floatX)
    X_ = X_.reshape( [X_.shape[0]*repeat, X.shape[1]] )
    return X_

def f_logsumexp(A, axis=None):
    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(T.sum(T.exp(A-A_max), axis=axis, keepdims=True))+A_max
    B = T.sum(B, axis=axis)
    return B


#-----------------------------------------------------------------------------
class STBPStack(Model):
    def __init__(self, **hyper_params):     
        super(STBPStack, self).__init__()

        # Hyper parameters
        self.register_hyper_param('layers', help='STBP layers', default=[])
        self.register_hyper_param('n_samples', help='no. of samples to use', default=100)

        self.set_hyper_params(hyper_params)

    def setup(self):
        layers = self.layers
        n_layers = len(layers)
        
        assert isinstance(layers[-1], STBPTop)

        for l in xrange(0, n_layers-1):
            assert isinstance(layers[l], STBPLayer)
            layers[l].n_upper = layers[l+1].n_lower
            layers[l].setup()

        self.n_lower = layers[0].n_lower

    def sample_p(self, n_samples):
        layers = self.layers
        n_layers = len(layers)

        # Generate samples from the generative model
        samples = [None]*n_layers

        samples[-1], log_p = layers[-1].sample_p(n_samples)
        for l in xrange(n_layers-1, 0, -1):
            samples[l-1], log_p_l = layers[l-1].sample_p(samples[l])
            log_p += log_p_l
        
        return samples[0], log_p

    def log_likelihood(self, X, Y=None, n_samples=None):
        layers = self.layers
        n_layers = len(layers)

        if n_samples == None:
            n_samples = self.n_samples

        batch_size = X.shape[0]

        # Prepare input for layers
        samples = [None]*n_layers
        log_q   = [None]*n_layers
        log_p   = [None]*n_layers

        samples[0] = f_replicate_batch(X, n_samples)                   # 
        log_q[0]   = T.zeros([batch_size*n_samples])+T.log(n_samples)  # 1/n_samples for each replicted X   XXX REALLY XXX
        
        # Generate samples (feed-forward)
        for l in xrange(n_layers-1):
            samples[l+1], log_q[l+1] = layers[l].sample_q(samples[l])
        
        # Get log_probs from generative model
        log_p[n_layers-1] = layers[n_layers-1].log_p(samples[n_layers-1])
        for l in xrange(n_layers-1, 0, -1):
            log_p[l-1] = layers[l-1].log_p(samples[l-1], samples[l])

        # Reshape and sum
        log_p_all = T.zeros((batch_size, n_samples))
        log_q_all = T.zeros((batch_size, n_samples))
        for l in xrange(n_layers):
            samples[l] = samples[l].reshape((batch_size, n_samples, layers[l].n_lower))
            log_q[l] = log_q[l].reshape((batch_size, n_samples))
            log_p[l] = log_p[l].reshape((batch_size, n_samples))

            log_p_all += log_p[l]   # agregate all layers
            log_q_all += log_q[l]   # agregate all layers

        # Approximate P(X)
        log_px = f_logsumexp(log_p_all-log_q_all, axis=1)
        
        # Calculate samplig weights
        w = T.exp(log_p_all-log_q_all-T.shape_padright(log_px))

        # Calculate KL(P|Q), Hp, Hq
        KL = [None]*n_layers
        Hp = [None]*n_layers
        Hq = [None]*n_layers
        for l in xrange(n_layers):
            KL[l] = T.sum(w*(log_p[l]-log_q[l]), axis=1)
            Hp[l] = T.sum(w*log_p[l], axis=1)
            Hq[l] = T.sum(w*log_q[l], axis=1)

        return log_px, w, log_p_all, log_q_all, KL, Hp, Hq

    def get_gradients(self, X, Y=None, lr_p=1., lr_q=1., n_samples=None):
        """ return log_PX and an OrderedDict with parameter gradients """
        log_PX, w, log_p, log_q, KL, Hp, Hq = self.log_likelihood(X, Y, n_samples=n_samples)
        
        batch_log_PX = T.sum(log_PX)
        cost_p = T.sum(T.sum(log_p*w, axis=1))
        cost_q = T.sum(T.sum(log_q*w, axis=1))

        gradients = OrderedDict()
        for nl, layer in enumerate(self.layers):
            for name, shvar in layer.get_p_params().iteritems():
                gradients[shvar] = lr_p[nl] * T.grad(cost_p, shvar, consider_constant=[w])

            for name, shvar in layer.get_q_params().iteritems():
                gradients[shvar] = lr_q[nl] * T.grad(cost_q, shvar, consider_constant=[w])

        return batch_log_PX, gradients

    #------------------------------------------------------------------------
    def get_p_params(self):
        params = OrderedDict()
        for l in self.layers:
            params.update( l.get_p_params() )
        return params

    def get_q_params(self):
        params = OrderedDict()
        for l in self.layers:
            params.update( l.get_q_params() )
        return params

    def model_params_to_dlog(self, dlog):
        vals = {}
        for n,l in enumerate(self.layers):
            for pname, shvar in l.get_p_params().iteritems():
                key = "L%d.P.%s" % (n, pname)
                vals[key] = shvar.get_value()
            for pname, shvar in l.get_q_params().iteritems():
                key = "L%d.Q.%s" % (n, pname)
                vals[key] = shvar.get_value()
        dlog.append_all(vals)

    def model_params_from_dlog(self, dlog, row=-1):
        for n,l in enumerate(self.layers):
            for pname, shvar in l.get_p_params().iteritems():
                key = "L%d.P.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)
            for pname, shvar in l.get_q_params().iteritems():
                key = "L%d.Q.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)
 

#-----------------------------------------------------------------------------
class STBPLayer(Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(STBPLayer, self).__init__()
        self.register_hyper_param('clamp_sigmoid', default=True)
        self.q_nade = None

    def sigmoid(self, x):
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    
    @abc.abstractmethod
    def setup(self):
        """ Setup a proper Q() model for this layer """
        pass

    def get_p_params(self):
        return self.get_model_params()
    
    def get_q_params(self):
        assert self.q_nade is not None
        return self.q_nade.get_model_params()
 
    #-------------------------------------------------------------------------
    @abc.abstractmethod
    def log_p(self, X, H):
        """ Calculate P(X|H) 
            return log(P(X|H))
        """
        pass
    
    @abc.abstractmethod
    def sample_p(self, H):
        """ Sample X ~ P(X|H) 
            return X, log(P(X|H))
        """
        pass

    #-------------------------------------------------------------------------
    @abc.abstractmethod
    def log_q(self, X, H):
        """ Calulate Q(H|X)
            Return log(P(Q(H|X)))
        """
        pass

    @abc.abstractmethod
    def sample_q(self, X):  
        """ Given a set of visible X, generate proposal H~Q(H|X)  
            return H, log(Q(H|X))
        """
        pass
    

class STBPTop(Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **hyper_params):
        super(STBPTop, self).__init__()
        self.register_hyper_param('clamp_sigmoid', default=True)

    def sigmoid(self, x):
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    def setup(self):
        pass

    def get_p_params(self):
        return self.get_model_params()
    
    def get_q_params(self):
        return OrderedDict()
    
    #-------------------------------------------------------------------------
    @abc.abstractmethod
    def log_p(self, X):
        """ Calculate P(H) 
            return log(P(X|H))
        """
        pass

    @abc.abstractmethod
    def sample_p(self, n_samples):
        """ Sample X ~ P(X) 
            return X, log(P(X))
        """
        pass

    #-------------------------------------------------------------------------
    def log_q(self, X, H):
        """ Should never be called """ 
        raise RuntimeError("Sombody called log_q on a TopLayer")

    def sample_q(self, X):  
        """ Should never be called """ 
        raise RuntimeError("Sombody called q_sample on a TopLayer")


#=============================================================================
class FactoizedBernoulliTop(STBPTop):
    def __init__(self, **hyper_params):
        super(FactoizedBernoulliTop, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_lower', help='no. binary variables')

        # Model parameters
        self.register_model_param('a', help='P sigmoid(a) prior', default=lambda: np.zeros(self.n_lower))

        self.set_hyper_params(hyper_params)

    def log_p(self, X):
        """ Calculate P(H) 
            return log(P(X|H))
        """
        n_lower, = self.get_hyper_params(['n_lower'])
        a, = self.get_model_params(['a'])

        # Calculate log-bernoulli
        p_X = self.sigmoid(a)
        post = X*T.log(p_X) + (1-X)*T.log(1-p_X)
        post = post.sum(axis=1)

        return post

    def sample_p(self, n_samples):
        """ Sample X ~ P(X) 
            return X, log(P(X))
        """
        n_lower, = self.get_hyper_params(['n_lower'])
        a, = self.get_model_params(['a'])

        # sample hiddens
        p_X = self.sigmoid(a)
        X = T.cast(theano_rng.uniform((n_samples, n_lower)) <= p_X, dtype=floatX)

        return X, self.log_p(X)


class FVSBTop(STBPTop):
    def __init__(self, **hyper_params):
        super(FVSBTop, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_units', help='no. binary variables')

        # Model parameters
        self.register_model_param('b', help='sigmoid(b)-bias ', default=lambda: np.zeros(self.n_units))
        self.register_model_param('W', help='weights (triangular)', default=lambda: default_weights(self.n_units, self.n_units) )

        self.set_hyper_params(hyper_params)

    def log_p(self, H):
        """ Calculate P(H) 
            return log(P(X|H))
        """
        n_units, = self.get_hyper_params(['n_units'])
        W, b = self.get_model_params(['W', 'b'])

        # Calculate log-bernoulli
        W   = tensor.tril(W, k=-1)
        p_i = self.sigmoid(T.dot(H, W)+b)

        post = H*T.log(p_i) + (1-H)*T.log(1-p_i)
        post = post.sum(axis=1)

        return post

    def sample_p(self, n_samples):
        """ Sample H ~ P(H) 
            return H, log(P(H))
        """
        n_units, = self.get_hyper_params(['n_units'])
        W, b = self.get_model_params(['W', 'b'])

        # Calculate log-bernoulli
        W   = tensor.tril(W, k=-1)
        p_i = self.sigmoid(T.dot(H, W)+b)

        post_init = T.zeros(n_samples, dtype=floatX)
        X_init    = T.zeros((n_samples, n_units), dtype=floatX)

        def one_iter():
            pass
    
        [X, post], updates = unroll_scan(
                    fn=one_iter, 
                    sequences=[W, b], 
                    outputs_info=[X_init, post_init]
                )
        assert len(updates) == 0


        # sample hiddens
        p_X = self.sigmoid(a)
        X = T.cast(theano_rng.uniform((n_samples, n_lower)) <= p_X, dtype=floatX)

        return X, self.log_p(X)



#=============================================================================
class SigmoidBeliefLayer(STBPLayer):
    def __init__(self, **hyper_params):
        super(SigmoidBeliefLayer, self).__init__()

        self.register_hyper_param('n_lower', help='no. lower-layer binary variables')
        self.register_hyper_param('n_upper', help='no. upper-layer binary variables')
        self.register_hyper_param('n_qhid',  help='no. CNADE latent binary variables', default=lambda: 2*self.n_upper)
        self.register_hyper_param('unroll_scan', default=1)

        # Sigmoid Belief Layer
        self.register_model_param('b', help='P lower-layer bias', default=lambda: np.zeros(self.n_lower))
        self.register_model_param('W', help='P weights', default=lambda: default_weights(self.n_upper, self.n_lower) )

        self.set_hyper_params(hyper_params)

    def setup(self):
        assert self.q_nade is None
        
        self.q_nade = CNADE(
                    n_vis=self.n_upper,
                    n_cond=self.n_lower,
                    n_hid=self.n_qhid,
                    clamp_sigmoid=self.clamp_sigmoid
                )

    #-------------------------------------------------------------------------
    def log_p(self, X, H):
        """ Calculate P(X|H) 
            return log(P(X|H))
        """
        W, b = self.get_model_params(['W', 'b'])

        # Posterior P(X|H)
        p_X = self.sigmoid(T.dot(H, W) + b)
        lp_X = X*T.log(p_X) + (1-X)*T.log(1-p_X)
        lp_X = T.sum(lp_X, axis=1)

        return lp_X
    
    def sample_p(self, H):
        """ Sample X ~ P(X|H) 
            return X, log(P(X|H))
        """
        n_lower, = self.get_hyper_params(['n_lower'])
        W, b = self.get_model_params(['W', 'b'])

        n_samples = H.shape[0]

        # sample X given H
        p_X = self.sigmoid(T.dot(H, W) + b)
        X = T.cast(theano_rng.uniform((n_samples, n_lower)) <= p_X, dtype=floatX)

        post = X*T.log(p_X) + (1-X)*T.log(1-p_X)
        post = post.sum(axis=1)

        return X, post

    #-------------------------------------------------------------------------
    def log_q(self, X, H):
        """ Calulate Q(H|X)
            Return log(P(Q(H|X)))
        """
        q_nade = self.q_nade
        return q_nade.f_loglikelihood(H, X)

    def sample_q(self, X):  
        """ Given a set of visible X, generate proposal H~Q(H|X)  
            return H, log(Q(H|X))
        """
        q_nade =  self.q_nade
        return q_nade.f_sample(X)


#-----------------------------------------------------------------------------
def get_toy_model():
    layers = [
        SigmoidBeliefLayer( 
            unroll_scan=1,
            n_lower=25,
            n_qhid=25,
        ),
        FactoizedBernoulliTop(
            n_lower=10,
        )
    ]
    model = STBPStack(
        layers=layers
    )
    return model



