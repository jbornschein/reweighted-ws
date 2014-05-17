#!/usr/bin/env python 

from __future__ import division

import logging
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

from model import Model, default_weights
from utils.unrolled_scan import unrolled_scan
from utils.datalog  import dlog

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

class FactoizedBernoulliTop(TopModule):
    """ FactoizedBernoulliTop top layer """
    def __init__(self, **hyper_params):
        super(FactoizedBernoulliTop, self).__init__()
        
        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')

        # Model parameters
        self.register_model_param('a', help='sigmoid(a) prior', 
            default=lambda: np.zeros(self.n_X))

        self.set_hyper_params(hyper_params)
    
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
        n_X, = self.get_hyper_params(['n_X'])
        a, = self.get_model_params(['a'])

        # sample hiddens
        prob_X = self.sigmoid(a)
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)        
        X = T.cast(U <= prob_X, dtype=floatX)

        return X, self.log_prob(X)

    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor 
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        a, = self.get_model_params(['a'])

        # Calculate log-bernoulli
        prob_X = self.sigmoid(a)
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

        return log_prob


class FVSBNTop(TopModule):
    def __init__(self, **hyper_params):
        super(ARSBNTop, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')

        # Model parameters
        self.register_model_param('b', help='sigmoid(b)-bias ', default=lambda: np.zeros(self.n_X))
        self.register_model_param('W', help='weights (triangular)', default=lambda: default_weights(self.n_X, self.n_X) )

        self.set_hyper_params(hyper_params)

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
        n_X, = self.get_hyper_params(['n_X'])
        W, b = self.get_model_params(['W', 'b'])

        # Calculate log-bernoulli
        W   = tensor.tril(W, k=-1)
        p_i = self.sigmoid(T.dot(H, W)+b)

        post_init = T.zeros(n_samples, dtype=floatX)
        X_init    = T.zeros((n_samples, n_X), dtype=floatX)

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
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)
        X = T.cast(U <= p_X, dtype=floatX)

        return X, self.log_p(X)

    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor 
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        W, b = self.get_model_params(['W', 'b'])

        # Calculate log-bernoulli
        W   = tensor.tril(W, k=-1)
        p_i = self.sigmoid(T.dot(H, W)+b)

        post = H*T.log(p_i) + (1-H)*T.log(1-p_i)
        post = post.sum(axis=1)

        return post


class NADE(TopModule):
    """ Top Level NADE """
    def __init__(self, **hyper_params):
        super(NADE, self).__init__()

        self.register_hyper_param('n_X', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('unroll_scan', default=1)

        self.register_model_param('b',  help='visible bias', default=lambda: np.zeros(self.n_X))
        self.register_model_param('c',  help='hidden bias' , default=lambda: np.zeros(self.n_hid))
        self.register_model_param('W',  help='encoder weights', default=lambda: default_weights(self.n_X, self.n_hid) )
        self.register_model_param('V',  help='decoder weights', default=lambda: default_weights(self.n_hid, self.n_X) )
        
        self.set_hyper_params(hyper_params)
   
    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor 
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, n_hid = self.get_hyper_params(['n_X', 'n_hid'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])
        
        batch_size = X.shape[0]
        vis = X

        #------------------------------------------------------------------
    
        a_init    = T.zeros([batch_size, n_hid]) + T.shape_padleft(c)
        post_init = T.zeros([batch_size], dtype=floatX)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return post[-1,:]

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
        n_X, n_hid = self.get_hyper_params(['n_X', 'n_hid'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])

        #------------------------------------------------------------------
    
        a_init    = T.zeros([n_samples, n_hid]) + T.shape_padleft(c)
        post_init = T.zeros([n_samples], dtype=floatX)
        vis_init  = T.zeros([n_samples], dtype=floatX)
        rand      = theano_rng.uniform((n_X, n_samples), nstreams=512)

        def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            vis_i = T.cast(rand_i <= pi, floatX)
            post  = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b, rand], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return vis.T, post[-1,:]

#----------------------------------------------------------------------------

class SigmoidBeliefLayer(Module):
    """ SigmoidBeliefLayer """
    def __init__(self, **hyper_params):
        super(SigmoidBeliefLayer, self).__init__()

        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')
        # self.register_hyper_param('n_qhid',  help='no. CNADE latent binary variables', default=lambda: 2*self.n_upper)
        self.register_hyper_param('unroll_scan', default=1)

        # Sigmoid Belief Layer
        self.register_model_param('b', help='P lower-layer bias', default=lambda: np.zeros(self.n_X))
        self.register_model_param('W', help='P weights', default=lambda: default_weights(self.n_Y, self.n_X) )

        self.set_hyper_params(hyper_params)

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
        n_X, = self.get_hyper_params(['n_X'])
        W, b = self.get_model_params(['W', 'b'])

        n_samples = Y.shape[0]

        # sample X given Y
        prob_X = self.sigmoid(T.dot(Y, W) + b)
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)
        X = T.cast(U <= prob_X, dtype=floatX)

        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

        return X, log_prob

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
        W, b = self.get_model_params(['W', 'b'])

        # posterior P(X|Y)
        prob_X = self.sigmoid(T.dot(Y, W) + b)
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = T.sum(log_prob, axis=1)

        return log_prob


class CNADE(Module):
    """ Conditional NADE """
    def __init__(self, **hyper_params):
        super(CNADE, self).__init__()

        self.register_hyper_param('n_X', help='no. observed binary variables')
        self.register_hyper_param('n_Y', help='no. conditioning binary variables')        
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('unroll_scan', default=1)

        self.register_model_param('b',  help='visible bias', default=lambda: np.zeros(self.n_X))
        self.register_model_param('c',  help='hidden bias' , default=lambda: np.zeros(self.n_hid))
        self.register_model_param('Ub', help='cond. weights Ub', default=lambda: default_weights(self.n_Y, self.n_X) )
        self.register_model_param('Uc', help='cond. weights Uc', default=lambda: default_weights(self.n_Y, self.n_hid) )
        self.register_model_param('W',  help='encoder weights', default=lambda: default_weights(self.n_X, self.n_hid) )
        self.register_model_param('V',  help='decoder weights', default=lambda: default_weights(self.n_hid, self.n_X) )
        
        self.set_hyper_params(hyper_params)
   
    def sample(self, Y):
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
        n_X, n_Y, n_hid = self.get_hyper_params(['n_X', 'n_Y', 'n_hid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])

        batch_size = Y.shape[0]
        cond = Y

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
    
        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=floatX)
        vis_init  = T.zeros([batch_size], dtype=floatX)
        rand      = theano_rng.uniform((n_X, batch_size), nstreams=512)

        def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            vis_i = T.cast(rand_i <= pi, floatX)
            post  = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
            a     = a + T.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b_cond.T, rand], 
                    outputs_info=[a_init, vis_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return vis.T, post[-1,:]

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
        n_X, n_Y, n_hid    = self.get_hyper_params(['n_X', 'n_Y', 'n_hid'])
        b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])
        
        batch_size = X.shape[0]
        vis = X
        cond = Y

        #------------------------------------------------------------------
        b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
        c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
    
        a_init    = c_cond
        post_init = T.zeros([batch_size], dtype=floatX)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = self.sigmoid(a)
            pi   = self.sigmoid(T.dot(hid, Vi) + bi)
            post = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b_cond.T],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        assert len(updates) == 0
        return post[-1,:]

#=============================================================================

class STBPStack(Model):
    def __init__(self, **hyper_params):     
        super(STBPStack, self).__init__()

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

            p_layers[l].setup()
            q_layers[l].setup()

            assert p_layers[l].n_Y == p_layers[l+1].n_X
            assert p_layers[l].n_Y == q_layers[l].n_X 

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

    def log_likelihood(self, X, Y=None, n_samples=None):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

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
            samples[l+1], log_q[l+1] = q_layers[l].sample(samples[l])
        
        # Get log_probs from generative model
        log_p[n_layers-1] = p_layers[n_layers-1].log_prob(samples[n_layers-1])
        for l in xrange(n_layers-1, 0, -1):
            log_p[l-1] = p_layers[l-1].log_prob(samples[l-1], samples[l])

        # Reshape and sum
        log_p_all = T.zeros((batch_size, n_samples))
        log_q_all = T.zeros((batch_size, n_samples))
        for l in xrange(n_layers):
            samples[l] = samples[l].reshape((batch_size, n_samples, p_layers[l].n_X))
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

    def get_gradients(self, X, Y, lr_p, lr_q, n_samples):
        """ return log_PX and an OrderedDict with parameter gradients """
        log_PX, w, log_p, log_q, KL, Hp, Hq = self.log_likelihood(X, Y, n_samples=n_samples)
        
        batch_log_PX = T.sum(log_PX)
        cost_p = T.sum(T.sum(log_p*w, axis=1))
        cost_q = T.sum(T.sum(log_q*w, axis=1))

        gradients = OrderedDict()
        for nl, layer in enumerate(self.p_layers):
            for name, shvar in layer.get_model_params().iteritems():
                gradients[shvar] = lr_p[nl] * T.grad(cost_p, shvar, consider_constant=[w])

        for nl, layer in enumerate(self.q_layers):
            for name, shvar in layer.get_model_params().iteritems():
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
            for name, shvar in layer.get_model_params().iteritems():
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
            for pname, shvar in l.get_model_params().iteritems():
                key = "L%d.P.%s" % (n, pname)
                vals[key] = shvar.get_value()
        for n,l in enumerate(self.q_layers):                
            for pname, shvar in l.get_model_params().iteritems():
                key = "L%d.Q.%s" % (n, pname)
                vals[key] = shvar.get_value()
        return vals

    def model_params_from_dict(self, vals):
        for n,l in enumerate(self.p_layers):
            for pname, shvar in l.get_model_params().iteritems():
                key = "L%d.P.%s" % (n, pname)
                value = vals[key]
                shvar.set_value(value)
        for n,l in enumerate(self.q_layers):                
            for pname, shvar in l.get_model_params().iteritems():
                key = "L%d.Q.%s" % (n, pname)
                value = vals[key]
                shvar.set_value(value)

    def model_params_to_dlog(self, dlog):
        vals = self.model_params_to_dict()
        dlog.append_all(vals)

    def model_params_from_dlog(self, dlog, row=-1):
        for n,l in enumerate(self.p_layers):
            for pname, shvar in l.get_model_params().iteritems():
                key = "L%d.P.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)
        for n,l in enumerate(self.q_layers):                
            for pname, shvar in l.get_model_params().iteritems():
                key = "L%d.Q.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)
 
#=============================================================================

def get_toy_model():
    p_layers = [
        SigmoidBeliefLayer( 
            n_X=25,
            n_Y=10
        ),
        FactoizedBernoulliTop(
            n_X=10,
        )
    ]
    q_layers = [
        CNADE(
            unroll_scan=1,
            n_X=10,
            n_Y=25,
            n_hid=10
        )
    ]
    model = STBPStack(
        p_layers=p_layers,
        q_layers=q_layers,
    )
    return model
