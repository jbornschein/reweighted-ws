#!/usr/bin/env python

import unittest

from learning.training import Trainer
from learning.tests.toys import *

def test_complete():
    t = Trainer(
        dataset=get_toy_data(),
        model=get_toy_model(),
    )

    t.load_data()
    t.compile()
    t.perform_epoch()
    
