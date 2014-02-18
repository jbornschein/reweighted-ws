#!/usr/bin/env python

from __future__ import division

import sys
sys.path.append("../lib")
# sys.setrecursionlimit(10000)

import logging
from time import time
import cPickle as pickle

import numpy as np

import theano
import theano.tensor as T

from datalog import dlog, StoreToH5, TextPrinter

from dataset import ToyData, MNIST
from training import BatchedSGD
from cnade import CNADE

_logger = logging.getLogger()

#=============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    _logger.info("setting up datalogger...")
    dlog.set_handler("*", StoreToH5, "mnist-experiment-dump.h5")
    dlog.set_handler(["L", "LL_epoch"], TextPrinter)

    _logger.info("loading data...")
    data_train = MNIST(which_set='train')
    data_valid = MNIST(which_set='valid')

    batch_size = 10
    learning_rate = 0.0005

    N, n_vis  = data_train.X.shape
    N, n_cond = data_train.Y.shape
    n_hid = 100

    _logger.info("instatiating model")
    nade = CNADE(n_vis=n_vis, n_hid=n_hid, n_cond=n_cond, batch_size=batch_size)

    _logger.info("instatiating trainer")
    trainer = BatchedSGD(batch_size=batch_size)
    trainer.set_data(data_train, data_valid)
    trainer.set_model(nade)
    trainer.compile()

    print "=" * 77
    epochs = 0
    end_learning = False

    LL = [-np.inf]
    t0 = time()
    while not end_learning:
        LL_epoch = 0.

        trainer.perform_epoch(learning_rate)

        # for b in xrange(N//batch_size):
        #    first = batch_size*b
        #    last  = first + batch_size
        #    batch_x = train_x[first:last]
        #
#            _, L, gb, gc, gW, gV = f_post(batch_x, learning_rate)

#            dlog.progress("Prcessing minibatch %d" % b, b/(N//batch_size))
#            dlog.append("L", L)
# print "gb:", gb
# print "b:", nade.b.get_value()
#
#            LL_epoch += L
#        LL_epoch = LL_epoch / (N//batch_size)
#        LL.append(LL_epoch)
#
#        dlog.append_all( {
#            "LL_epoch": LL_epoch,
#            "b": nade.b.get_value(),
#            "c": nade.c.get_value(),
#            "W": nade.W.get_value(),
#            "V": nade.V.get_value()
#        })
#
#        if epochs % 1 == 0:
#            print "--- %d ----" % epochs
#            print "LL:", LL_epoch
# print "post: ", post
# print "gb:", gb
# print "b:", nade.b.get_value()
# print "c:", nade.c.get_value()
#        epochs += 1
#
# Converged?
#        end_learning = LL_epoch <= np.max(LL[-6:-1])
#        end_learning |= epochs > 10000
    t = time() - t0
    print "Time per epoch: %f" % (t / epochs)
