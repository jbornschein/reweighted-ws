import unittest 

import numpy as np

# Unit Under Test
from learning.hyperbase import * 

class ExampleThing(HyperBase):
    def __init__(self, **hyper_params):
        super(ExampleThing, self).__init__(**hyper_params)

        self.register_hyper_param("hyper_a")
        self.register_hyper_param("hyper_b", default=23)
        self.register_hyper_param("hyper_c", default=lambda: 2*21, help="help")

        self.set_hyper_params(hyper_params)


def test_constructor():
    model = ExampleThing(hyper_a=0)

    hyper_a = model.get_hyper_param('hyper_a')
    assert hyper_a == 0

def test_hyper_defaults():
    model = ExampleThing()

    assert model.get_hyper_param('hyper_b') == 23, model.get_hyper_param('hyper_b')
    assert model.get_hyper_param('hyper_c') == 42, model.get_hyper_param('hyper_c')

def test_hyper_setget():
    model = ExampleThing()

    model.set_hyper_param('hyper_b', 1)
    model.set_hyper_param('hyper_c', 2)
    assert model.get_hyper_param('hyper_b') == 1
    assert model.get_hyper_params(['hyper_b', 'hyper_c']) == [1, 2]

    model.set_hyper_params({'hyper_b': 23, 'hyper_c': 42})
    assert model.get_hyper_param('hyper_b') == 23
    assert model.get_hyper_params(['hyper_b', 'hyper_c']) == [23, 42]

def test_hyper_attr():
    model = ExampleThing()

    
    assert model.hyper_b == 23
    assert model.hyper_c == 42

    model.hyper_a = 11
    assert model.hyper_a == 11

