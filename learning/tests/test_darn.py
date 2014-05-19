import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing
from test_isws import ISLayerTest, ISTopLayerTest

# Unit Under Test
from learning.darn import DARN, DARNTop


#-----------------------------------------------------------------------------

class TestDARN(ISTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DARNTop(
                        n_X=8,
                    )
        self.layer.setup()

class TestDARN(ISLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DARN(
                        n_X=16,
                        n_Y=8,
                    )
        self.layer.setup()

