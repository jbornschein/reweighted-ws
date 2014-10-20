import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing
from test_rws import RWSLayerTest, RWSTopLayerTest

# Unit Under Test
from learning.dsbn import DSBN

#-----------------------------------------------------------------------------

class TestDSBN(RWSLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DSBN(
                        n_X=16,
                        n_Y=8,
                        n_hid=12,
                    )
        self.layer.setup()
