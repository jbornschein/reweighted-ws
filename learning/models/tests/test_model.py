import unittest 

import numpy as np

from collections import OrderedDict

# Unit Under Test
from learning.model import * 

class ExampleModel(Model):
    def __init__(self, **hyper_params):
        super(ExampleModel, self).__init__(**hyper_params)

        self.register_hyper_param("hyper_a")
        self.register_hyper_param("hyper_b", default=23)
        self.register_hyper_param("hyper_c", default=lambda: 2*21, help="help")

        self.register_model_param("model_a")
        self.register_model_param("model_b")
        self.register_model_param("model_c", help="help")
    
        self.set_hyper_params(hyper_params)


def test_constructor():
    model = ExampleModel(hyper_a=0)

    hyper_a = model.get_hyper_param('hyper_a')
    assert hyper_a == 0

def test_hyper_defaults():
    model = ExampleModel()

    assert model.get_hyper_param('hyper_b') == 23, model.get_hyper_param('hyper_b')
    assert model.get_hyper_param('hyper_c') == 42, model.get_hyper_param('hyper_c')

def test_hyper_setget():
    model = ExampleModel()

    model.set_hyper_param('hyper_b', 1)
    model.set_hyper_param('hyper_c', 2)
    assert model.get_hyper_param('hyper_b') == 1
    assert model.get_hyper_params(['hyper_b', 'hyper_c']) == [1, 2]

    model.set_hyper_params({'hyper_b': 23, 'hyper_c': 42})
    assert model.get_hyper_param('hyper_b') == 23
    assert model.get_hyper_params(['hyper_b', 'hyper_c']) == [23, 42]

def test_model_setget():
    model = ExampleModel()

    model.set_model_param('model_b', 1)
    model.set_model_param('model_c', 2)
    assert model.get_model_param('model_b').get_value() == 1
    assert model.get_model_param('model_c').get_value() == 2

    model.set_model_params({'model_b': 23, 'model_c': 42})
    assert model.get_model_param('model_b').get_value() == 23
    #assert model.get_model_params(['model_b', 'model_c']) == [23, 42]

def test_get_all_model_params():
    model = ExampleModel()

    model.set_model_param('model_a', 1)
    all_params = model.get_model_params()
   
    assert type(all_params) == OrderedDict
    assert len(all_params) == 3

def test_hyper_attr():
    model = ExampleModel()

    
    assert model.hyper_b == 23
    assert model.hyper_c == 42

    model.hyper_a = 11
    assert model.hyper_a == 11

def test_model_attr():
    model = ExampleModel()

    model.model_a = 23.5
    assert np.allclose(model.model_a.get_value(), 23.5)
