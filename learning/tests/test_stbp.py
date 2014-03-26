import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing

# Unit Under Test
from learning.stbp_layers import *

#-----------------------------------------------------------------------------

class STBPLayerTest(object):

    def test_basic_log_p(self):
        n_samples = self.n_samples
        layer = self.layer

        X, X_ = testing.fmatrix( (n_samples, layer.n_lower), name="X")
        H, H_ = testing.fmatrix( (n_samples, layer.n_upper), name="H")

        log_p = layer.log_p(X, H)
        do_log_p = theano.function([X, H], log_p, name="log_p")

        log_p_ = do_log_p(X_, H_)
        assert log_p_.shape == (n_samples,)


    def test_basic_sample_p(self):
        n_samples = self.n_samples
        layer = self.layer

        H, H_ = testing.fmatrix( (n_samples, layer.n_upper), name="H")

        X, log_p = layer.sample_p(H)
        do_sample_p = theano.function([H], [X, log_p], name="sample_p")
        
        X_, log_p_ = do_sample_p(H_)
        assert X_.shape == (n_samples, layer.n_lower )
        assert log_p_.shape == (n_samples,)
 
    #------------------------------------------------------------------------
    def basic_log_q(self):
        n_samples = self.n_samples
        layer = self.layer
        pass

    def test_basic_sample_q(self):
        n_samples = self.n_samples
        layer = self.layer
     
        X, X_ = testing.fmatrix( (n_samples, layer.n_lower), "X")
        H, log_q = layer.sample_q(X)

        do_sample_q = theano.function([X], [H, log_q], name="sample_q")

        H_, log_q_ = do_sample_q(X_)

        assert H_.shape == (n_samples, layer.n_upper)
        assert log_q_.shape == (n_samples,)


       
#    def test_dPdTheta(self):
#        n_samples = self.n_samples
#        layer = self.layer
#
#        X, log_p = layer.sample_p(H)
#        
#
#
#        updates = OrderedDict()
#        for name,shvar in layer.get_p_params().iteritems():
#            print "Taking gradient d P(X, H) / d%s" % name
#            updates[shvar] = T.grad(log_p
            
class STBPTopLayerTest(object):
    def test_basic_log_p(self):
        n_samples = self.n_samples
        layer = self.layer

        X, X_ = testing.fmatrix( (n_samples, layer.n_lower), name="X")

        log_p = layer.log_p(X)
        do_log_p = theano.function([X], log_p, name="log_p")

        log_p_ = do_log_p(X_)
        assert log_p_.shape == (n_samples,)
 
    def test_basic_sample_p(self):
        n_samples = self.n_samples
        layer = self.layer

        X, log_p = layer.sample_p(n_samples)
        do_sample_p = theano.function([], [X, log_p], name="sample_p")
        
        X_, log_p_ = do_sample_p()
        assert X_.shape == (n_samples, layer.n_lower )
        assert log_p_.shape == (n_samples,)
 

        

#-----------------------------------------------------------------------------

class TestFactorizedBernoulliTop(STBPTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = FactoizedBernoulliTop(
                        clamp_sigmoid=True,
                        n_lower=8
                    )
        self.layer.setup()

class TestSigmoidBeliefLayer(STBPLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = SigmoidBeliefLayer(
                        clamp_sigmoid=True,
                        n_lower=16,
                        n_upper=8,
                    )
        self.layer.setup()

#-----------------------------------------------------------------------------
# Test full STBTStack

class TestSTBTStack(unittest.TestCase):
    n_samples = 25
    n_vis = 8
    n_hid = 16
    n_qhid = 32

    def setUp(self):
        layers=[
            SigmoidBeliefLayer( 
                unroll_scan=1,
                n_lower=self.n_vis,
                n_qhid=self.n_qhid,
            ),
            SigmoidBeliefLayer( 
                unroll_scan=1,
                n_lower=self.n_hid,
                n_qhid=self.n_qhid,
            ),
            FactoizedBernoulliTop(
                n_lower=self.n_hid
            )
        ]
        self.stack = STBPStack(layers=layers)
        self.stack.setup()

    def test_layer_sizes(self):
        stack = self.stack
        layers = stack.layers
        n_layers = len(layers)

        for l in xrange(n_layers-1):
            assert layers[l].n_upper == layers[l+1].n_lower

    def test_sample_p(self):
        stack = self.stack
    
        n_samples, n_samples_ = testing.iscalar('n_samples')
        X, log_P = stack.sample_p(n_samples=n_samples)
        do_sample = theano.function([n_samples], [X[0], log_P], name="do_sample")

        X0_, log_P_ = do_sample(n_samples_)
    
        assert X0_.shape == (n_samples_, self.n_vis)
        assert log_P_.shape == (n_samples_, )

    def test_log_likelihood(self):
        batch_size = 20
        stack = self.stack
    
        X, X_ = testing.fmatrix((batch_size, self.n_vis), 'X')
        n_samples, n_samples_ = testing.iscalar('n_samples')
        n_samples_ = self.n_samples

        log_PX, w, log_P, log_Q, KL, Hp, Hq = stack.log_likelihood(X, n_samples=n_samples)
        do_log_likelihood = theano.function(
                                [X, n_samples],
                                [log_PX, log_P, log_Q, w]  
                            )
    
        log_PX_, log_P_, log_Q_, w_ = do_log_likelihood(X_, n_samples_)

        print "log_P.shape", log_P_.shape
        print "log_Q.shape", log_Q_.shape
        print "log_PX.shape", log_PX_.shape
        print "w.shape", w_.shape

        assert log_PX_.shape == (batch_size,)
        assert log_P_.shape == (batch_size, n_samples_)
        assert log_Q_.shape == (batch_size, n_samples_)
        assert w_.shape == (batch_size, n_samples_)

        n_layers = len(stack.layers)

        assert len(KL) == n_layers
        assert len(Hp) == n_layers
        assert len(Hq) == n_layers

    def test_ll_grad(self):
        
        learning_rate = 1e-3
        batch_size = 20
        stack = self.stack
    
        X, X_ = testing.fmatrix((batch_size, self.n_vis), 'X')
        n_samples, n_samples_ = testing.iscalar('n_samples')
        n_samples_ = self.n_samples

        log_PX, w, log_P, log_Q, KL, Hp, Hq = stack.log_likelihood(X, n_samples=n_samples)

        cost_p = T.sum(T.sum(log_P*w, axis=1))
        cost_q = T.sum(T.sum(log_Q*w, axis=1))

        updates = OrderedDict()
        for pname, shvar in stack.get_p_params().iteritems():
            print "Calculating gradient dP/d%s" % pname
            updates[shvar] = T.grad(cost_p, shvar, consider_constant=[w])

        for pname, shvar in stack.get_q_params().iteritems():
            print "Calculating gradient dQ/d%s" % pname
            updates[shvar] = T.grad(cost_q, shvar, consider_constant=[w])


        do_sgd_step = theano.function(
                                inputs=[X, n_samples],
                                outputs=[log_PX, cost_p, cost_q],
                                updates=updates,
                                name="sgd_step",
                            )
    
        log_PX_, cost_p_, cost_q_, = do_sgd_step(X_, n_samples_)

        assert log_PX_.shape == (batch_size,)

#-----------------------------------------------------------------------------

def test_replicat():
    Av = np.array( [[1., 2., 3.], 
                    [2., 3., 4.]]).astype(np.float32)
    A = T.fmatrix('A')
    A.tag.test_value = Av

    B = f_replicate_batch(A, 10)
    do_replicate = theano.function([A], B, name="f_replicate", allow_input_downcast=True)
    Bv = do_replicate(Av)
    
    assert Bv.shape == (20, 3)
    assert Bv[0 , 0] == 1.
    assert Bv[10, 0] == 2.
 
