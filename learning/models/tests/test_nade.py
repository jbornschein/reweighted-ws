import unittest 

import numpy as np

import theano 
import theano.tensor as T

from test_rws import RWSLayerTest, RWSTopLayerTest

# Unit Under Test
from learning.models.nade import NADE, NADETop


#-----------------------------------------------------------------------------

class TestNADETop(RWSTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = NADETop(
                        n_X=8,
                        n_hid=8,
                    )
        self.layer.setup()

class TestNADE(RWSLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = NADE(
                        n_X=16,
                        n_Y=8,
                        n_hid=8,
                    )
        self.layer.setup()
