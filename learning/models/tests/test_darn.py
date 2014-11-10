import unittest 

import numpy as np

import theano 
import theano.tensor as T

import learning.tests as testing
from test_rws import RWSLayerTest, RWSTopLayerTest

# Unit Under Test
from learning.models.darn import DARN, DARNTop


#-----------------------------------------------------------------------------

class TestDARNTop(RWSTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DARNTop(
                        n_X=8,
                    )
        self.layer.setup()

    def test_scan_vs_fast(self):
        n_samples = self.n_samples
        layer = self.layer

        X, X_ = testing.fmatrix( (n_samples, layer.n_X), name="X")

        log_prob = layer.log_prob(X)
        log_prob_scan = layer.log_prob(X)

        do_log_prob = theano.function([X], [log_prob, log_prob_scan], name="log_prob")

        log_prob_, log_prob_scan_ = do_log_prob(X_)

        assert np.allclose(log_prob_, log_prob_scan_)


class TestDARN(RWSLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DARN(
                        n_X=16,
                        n_Y=8,
                    )
        self.layer.setup()

