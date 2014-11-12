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


class TestDARN(RWSLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DARN(
                        n_X=16,
                        n_Y=8,
                    )
        self.layer.setup()

