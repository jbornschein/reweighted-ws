#!/usr/bin/env python 

from __future__ import division

import sys
sys.path.append("../lib")
sys.setrecursionlimit(10000)

import logging
from collections import OrderedDict 
from time import time
import gzip
import cPickle as pickle

import numpy as np

import theano 
import theano.tensor as T

from datalog import dlog, StoreToH5, TextPrinter
from nade import NADE

_logger = logging.getLogger("nade.py")


def preprocess(x, l):
    N = x.shape[0]
    assert N == l.shape[0]
    
    perm = np.random.permutation(N)
    x = x[perm,:]
    l = l[perm,:]

    x = 1*(x > 0.5)       # binarize x

    one_hot = np.zeros( (N, 10), dtype="float32")
    for n in xrange(N):
        one_hot[n, l[n]] = 1.
    
    return x, l
    
def permute(x, idx=None):
    if isinstance(x, list) or isinstance(x, tuple):
        if idx is None:
            _, n_vis = x[0].shape
        idx = np.random.permutation(n_vis)
        return [permute(i, idx) for i in x]
    
    if idx is None:
        _, n_vis = x.shape
        idx = np.random.permutation(n_vis)
    return x[:,idx]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    _logger.info("Setting up datalogger...")
    dlog.set_handler("*", StoreToH5, "mnist-experiment5.h5")
    dlog.set_handler(["L", "LL_epoch"], TextPrinter)

    _logger.info("Loading data...")
    with gzip.open("mnist.pkl.gz") as f:
        (train_x, train_l), (valid_x, valid_l), (test_x, test_l) = pickle.load(f)

    train_x, train_l = preprocess(train_x, train_l)
    valid_x, valid_l = preprocess(valid_x, valid_l)
    test_x , test_l  = preprocess(test_x, test_l)

    dlog.append("train_x", train_x[:1000])
    #train_x, valid_x, test_x = permute( [train_x, valid_x, test_x] )

    batch_size = 10
    N, n_vis = train_x.shape
    n_hid = 200

    nade = NADE(n_vis, n_hid, batch_size=batch_size)
    f_sample = nade.f_sample()
    f_post = nade.f_post()

    print "="*77
    epochs = 0
    learning_rate = 0.0005
    end_learning = False
    LL = [-np.inf]
    t0 = time()
    while not end_learning:
        LL_epoch = 0.
        for b in xrange(N//batch_size):
            first = batch_size*b
            last  = first + batch_size
            batch_x = train_x[first:last]

            _, L, gb, gc, gW, gV = f_post(batch_x, learning_rate)
        
            dlog.progress("Prcessing minibatch %d" % b, b/(N//batch_size))
            dlog.append("L", L)
            #print "gb:", gb
            #print "b:", nade.b.get_value()
            
            LL_epoch += L
        LL_epoch = LL_epoch / (N//batch_size)
        LL.append(LL_epoch)

        dlog.append_all( {
            "LL_epoch": LL_epoch, 
            "b": nade.b.get_value(),
            "c": nade.c.get_value(),
            "W": nade.W.get_value(),
            "V": nade.V.get_value()
        })

        if epochs % 1 == 0:
            print "--- %d ----" % epochs
            print "LL:", LL_epoch
            #print "post: ", post
            #print "gb:", gb
            #print "b:", nade.b.get_value()
            #print "c:", nade.c.get_value()
        epochs += 1
    
        # Converged?
        end_learning = LL_epoch <= np.max(LL[-6:-1]) 
        end_learning |= epochs > 10000
    t = time()-t0
    print "Time per epoch: %f" % (t/epochs)
    



