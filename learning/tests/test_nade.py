import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing
from test_isws import ISLayerTest, ISTopLayerTest

# Unit Under Test
from learning.nade import NADE, NADETop


#-----------------------------------------------------------------------------

class TestNADETop(ISTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = NADETop(
                        n_X=8,
                        n_hid=8,
                    )
        self.layer.setup()

class TestNADE(ISLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = NADE(
                        n_X=16,
                        n_Y=8,
                        n_hid=8,
                    )
        self.layer.setup()
