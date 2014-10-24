import unittest 

import numpy as np

import theano 
import theano.tensor as T

from test_rws import RWSLayerTest, RWSTopLayerTest

# Unit Under Test
from learning.models.sbn import SBN, SBNTop

#-----------------------------------------------------------------------------

class TestSBNTop(RWSTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = SBNTop(
                        n_X=8
                    )
        self.layer.setup()

class TestSBN(RWSLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = SBN(
                        n_X=16,
                        n_Y=8,
                    )
        self.layer.setup()
