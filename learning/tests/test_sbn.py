import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing
from test_isws import ISLayerTest, ISTopLayerTest

# Unit Under Test
from learning.sbn import SBN, SBNTop


#-----------------------------------------------------------------------------

class TestSBNTop(ISTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = SBNTop(
                        n_X=8
                    )
        self.layer.setup()

class TestSBN(ISLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = SBN(
                        n_X=16,
                        n_Y=8,
                    )
        self.layer.setup()
