#!/usr/bin/env python

"""
    Testcases for the pulp.utils package
"""

import os.path
import tempfile
import shutil
import numpy as np
import unittest

import autotable

#=============================================================================
# AutoTable tests

class TestAutotable(unittest.TestCase):
    def setUp(self):
        self.dirname = tempfile.mkdtemp()
        self.fname = os.path.join(self.dirname, "autotable-test.h5")
        self.at = autotable.AutoTable(self.fname)

    def tearDown(self):
        self.at.close()
        shutil.rmtree(self.dirname)

    def test_float(self):
        vals = {'testFloat': 2.42} 
        self.at.append_all( vals )

    def test_nparray(self):
        a    = np.ones( (10,10) )
        vals = {'testNPArray': a} 
        self.at.append_all( vals )
        
    def test_wrongShape(self):
        a    = np.ones( (10,10) )
        self.at.append_all( {'testWrongType': a} )

        b    = np.ones( (10) )
        self.assertRaises(TypeError, lambda: self.at.append_all( {'testWrongType': b} ) )

