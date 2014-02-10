#!/usr/bin/env python 

import sys
sys.path.append("../lib")

import numpy as np

import theano 
import theano.tensor as T


class NADE:
    def __init__(self, n_vis, n_hid):
        self.n_vis = n_vis
        self.n_hod = n_his

        W = np.zeros( (n_hid, n_vis), dtype=np.float)
        V = np.zeros( (n_vis, n_hid), dtype=np.float)

    def sample()
