#!/usr/bin/env python

import unittest

from learning.training import Trainer
from learning.dataset import get_toy_data
from learning.stbp_layers import get_toy_model

def test_complete():
    t = Trainer()
    t.dataset = get_toy_data()
    t.model = get_toy_model()

    t.load_data()
    t.compile()
    t.perform_epoch()
    
