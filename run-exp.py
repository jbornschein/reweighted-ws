#!/usr/bin/env python

from __future__ import division

import sys
# sys.setrecursionlimit(10000)

import logging
from time import time
import cPickle as pickle

import numpy as np

import theano
import theano.tensor as T

from learning.utils.datalog import dlog, StoreToH5, TextPrinter
from learning.experiment import Experiment

_logger = logging.getLogger()

#=============================================================================
if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('param_file')
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO)

    print "="*77
    print "== Starting experiment: %s" % args.param_file

    experiment = Experiment.from_param_file(args.param_file)
    experiment.setup_output_dir(args.param_file)
    experiment.setup_logging()
    experiment.setup_experiment()

    print "=="
    
    experiment.run_experiment()


    exit(0)

    _logger.info("loading data...")
    data_train = ToyData(which_set='train')
    data_valid = ToyData(which_set='valid')

    batch_size = 10
    learning_rate = 0.5

    N, n_vis = data_train.X.shape
    n_hid = 20

    print data_train.X.shape

    _logger.info("instatiating model")
    nade = NADE(n_vis=n_vis, n_hid=n_hid, batch_size=batch_size)

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
#    t = time() - t0
#    print "Time per epoch: %f" % (t / epochs)
