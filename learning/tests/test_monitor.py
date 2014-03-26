import unittest 

import numpy as np

import theano 
import theano.tensor as T

from learning.dataset import ToyData

# Unit Under Test
from learning.monitor import * 

def test_MonitorLL():
    dataset = ToyData('valid')
    n_samples = (1, 5, 25, 100, 500)
    monitor = MonitorLL(dataset, n_samples)



