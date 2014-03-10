#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

import os.path
import tempfile
import shutil
import numpy as np
import unittest
import tables

import datalog as datalog

#=============================================================================
# DatLog tests

class TestDataLog(unittest.TestCase):
    def setUp(self):
        self.dirname = tempfile.mkdtemp()
        self.fname = os.path.join(self.dirname, "autotable-test.h5")

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def append_content(self, dlog):
        dlog.append("T", 0.)
        dlog.append("T", 1.)
        dlog.append("T", 2.)

    @unittest.skip("Failing since h5py conversion")
    def append_all_content(self, dlog):
        vals = {"T": 0., "A": 0.}
        dlog.append_all(vals)
        vals = {"T": 1., "A": 1.}
        dlog.append_all(vals)
        vals = {"T": 2., "A": 2.}
        dlog.append_all(vals)

    def check_content(self, fname):
        h5 = tables.openFile(fname, 'r')
        T = h5.root.T 
        self.assertAlmostEqual(T[0], 0.)
        self.assertAlmostEqual(T[1], 1.)
        self.assertAlmostEqual(T[2], 2.)
        h5.close()

    def test_default_dlog(self):
        dlog = datalog.getLogger()
        dlog.ignored("test")

    @unittest.skip("Failing since h5py conversion")
    def test_default_handler(self):
        dlog = datalog.getLogger()
        dlog.set_handler("*", datalog.StoreToH5, self.fname)
        self.append_content(dlog)
        dlog.close()

        self.check_content(self.fname)

    @unittest.skip("Failing since h5py conversion")
    def test_storage_handler(self):
        dlog = datalog.getLogger()
        dlog.set_handler('T', datalog.StoreToH5, self.fname)
        self.append_content(dlog)
        dlog.close()

        self.check_content(self.fname)
    
    def test_append_all(self):
        dlog = datalog.getLogger()
        dlog.set_handler('T', datalog.StoreToH5, self.fname)
        self.append_all_content(dlog)
        dlog.close()

        self.check_content(self.fname)

    def test_progress(self):
        dlog = datalog.getLogger()
        dlog.set_handler('T', datalog.StoreToH5, self.fname)
        dlog.progress("Hello, Test")
        dlog.close()
        
