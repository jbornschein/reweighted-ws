import unittest 

import numpy as np


# Unit Under Test
from model import * 


class ExampleModel(Model):
    def __init__(self, hyper_params):
        self.register_hyper_param("hyper_a")
        self.register_hyper_param("hyper_b", default=23)
        self.register_hyper_param("hyper_c", default=42, help="help")

        self.register_model_param("model_a")
        self.register_model_param("model_b")
        self.register_model_param("model_c", help="help")
    
        super(ExampleModel, self).__init__(hyper_params)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = ExampleModel( {'hyper_a': 0} )

    def test_constructor_hyper(self):
        hyper_a = self.model.get_hyper_param('hyper_a')
        assert hyper_a == 0

    def test_defaults(self):
        hyper_b, hyper_c = self.model.get_hyper_param(['hyper_b', 'hyper_c'])
        assert hyper_b == 23
        assert hyper_c == 42
        
