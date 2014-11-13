import unittest 

import numpy as np

import theano 
import theano.tensor as T

import learning.tests as testing

# Unit Under Test
from learning.models.rws import *
from learning.models.sbn import SBN, SBNTop

#-----------------------------------------------------------------------------

class RWSTopLayerTest(object):
    def test_basic_log_prob(self):
        n_samples = self.n_samples
        layer = self.layer

        X, X_ = testing.fmatrix( (n_samples, layer.n_X), name="X")

        log_prob = layer.log_prob(X)
        do_log_prob = theano.function([X], log_prob, name="log_prob")

        log_prob_ = do_log_prob(X_)
        assert log_prob_.shape == (n_samples,)
        assert not np.isnan(log_prob_).any()
 
    def test_basic_sample(self):
        n_samples = self.n_samples
        layer = self.layer

        X, log_prob = layer.sample(n_samples)
        do_sample_p = theano.function([], [X, log_prob], name="sample")
        
        X_, log_prob_ = do_sample_p()
        assert X_.shape == (n_samples, layer.n_X )
        assert log_prob_.shape == (n_samples,)
        assert not np.isnan(log_prob_).any()

    def test_sample_expected(self):
        n_samples = self.n_samples
        layer = self.layer

        if getattr(layer, "sample_expected", None) is None:
            raise unittest.SkipTest("sample_expected not implemented")

        X, log_prob = layer.sample_expected(n_samples)
        do_sample_p = theano.function([], [X, log_prob], name="sample_expected")
        
        X_, log_prob_ = do_sample_p()
        assert X_.shape == (n_samples, layer.n_X )
        assert log_prob_.shape == (n_samples,)
        assert not np.isnan(log_prob_).any()



class RWSLayerTest(object):
    def test_basic_log_prob(self):
        n_samples = self.n_samples
        layer = self.layer

        X, X_ = testing.fmatrix( (n_samples, layer.n_X), name="X")
        Y, Y_ = testing.fmatrix( (n_samples, layer.n_Y), name="H")

        log_prob = layer.log_prob(X, Y)
        do_log_prob = theano.function([X, Y], log_prob, name="log_prob")

        log_prob_ = do_log_prob(X_, Y_)
        assert log_prob_.shape == (n_samples,)
        assert not np.isnan(log_prob_).any()


    def test_basic_sample(self):
        n_samples = self.n_samples
        layer = self.layer

        Y, Y_ = testing.fmatrix( (n_samples, layer.n_Y), name="Y")

        X, log_prob = layer.sample(Y)
        do_sample_p = theano.function([Y], [X, log_prob], name="sample")
        
        X_, log_prob_ = do_sample_p(Y_)
        assert X_.shape == (n_samples, layer.n_X )
        assert log_prob_.shape == (n_samples,)
        assert not np.isnan(log_prob_).any()


    def test_sample_expected(self):
        n_samples = self.n_samples
        layer = self.layer

        if getattr(layer, "sample_expected", None) is None:
            raise unittest.SkipTest("sample_expected not implemented")

        Y, Y_ = testing.fmatrix( (n_samples, layer.n_Y), name="Y")

        X, log_prob = layer.sample(Y)
        do_sample_p = theano.function([Y], [X, log_prob], name="sample")
        
        X_, log_prob_ = do_sample_p(Y_)
        assert X_.shape == (n_samples, layer.n_X )
        assert log_prob_.shape == (n_samples,)
        assert not np.isnan(log_prob_).any()


#-----------------------------------------------------------------------------

class TestLayerStack(unittest.TestCase):
    n_samples = 25
    n_vis = 8
    n_hid = 16
    n_qhid = 32

    def setUp(self):
        p_layers=[
            SBN( 
                n_X=self.n_vis,
                n_Y=self.n_hid,
            ),
            SBN( 
                n_X=self.n_hid,
                n_Y=self.n_hid,
            ),
            SBNTop(
                n_X=self.n_hid
            )
        ]
        q_layers=[
            SBN(
                n_Y=self.n_vis,
                n_X=self.n_hid,
            ),
            SBN(
                n_Y=self.n_hid,
                n_X=self.n_hid,
            )
        ]
        self.stack = LayerStack(p_layers=p_layers, q_layers=q_layers)
        self.stack.setup()

    def test_layer_sizes(self):
        stack = self.stack
        p_layers = stack.p_layers
        n_layers = len(p_layers)

        for l in xrange(n_layers-1):
            assert p_layers[l].n_Y == p_layers[l+1].n_X

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

        n_layers = len(stack.p_layers)

        assert len(KL) == n_layers
        assert len(Hp) == n_layers
        assert len(Hq) == n_layers


    def test_gradients(self):
        batch_size = 20
        stack = self.stack
        n_layers = len(stack.p_layers)
    
        X, X_ = testing.fmatrix((batch_size, self.n_vis), 'X')
        n_samples, n_samples_ = testing.iscalar('n_samples')
        n_samples_ = self.n_samples

        lr_p = np.ones(n_layers)
        lr_q = np.ones(n_layers)

        log_PX, gradients = stack.get_gradients(X, None, 
                    lr_p=lr_p, lr_q=lr_q, n_samples=n_samples_)

    def test_sleep_gradients(self):
        pass

    # def test_ll_grad(self):
        
    #     learning_rate = 1e-3
    #     batch_size = 20
    #     stack = self.stack
    
    #     X, X_ = testing.fmatrix((batch_size, self.n_vis), 'X')
    #     n_samples, n_samples_ = testing.iscalar('n_samples')
    #     n_samples_ = self.n_samples

    #     log_PX, w, log_P, log_Q, KL, Hp, Hq = stack.log_likelihood(X, n_samples=n_samples)

    #     cost_p = T.sum(T.sum(log_P*w, axis=1))
    #     cost_q = T.sum(T.sum(log_Q*w, axis=1))

    #     updates = OrderedDict()
    #     for pname, shvar in stack.get_p_params().iteritems():
    #         print "Calculating gradient dP/d%s" % pname
    #         updates[shvar] = T.grad(cost_p, shvar, consider_constant=[w])

    #     for pname, shvar in stack.get_q_params().iteritems():
    #         print "Calculating gradient dQ/d%s" % pname
    #         updates[shvar] = T.grad(cost_q, shvar, consider_constant=[w])


    #     do_sgd_step = theano.function(
    #                             inputs=[X, n_samples],
    #                             outputs=[log_PX, cost_p, cost_q],
    #                             updates=updates,
    #                             name="sgd_step",
    #                         )
    
    #     log_PX_, cost_p_, cost_q_, = do_sgd_step(X_, n_samples_)

    #     assert log_PX_.shape == (batch_size,)

#-----------------------------------------------------------------------------

def test_replicate_batch():
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
 
