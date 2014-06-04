#!/usr/bin/env python 

from __future__ import division, print_function

import sys
sys.path.append("../")

import os
import os.path
import logging
from time import time
import cPickle as pickle

import numpy as np

logger = logging.getLogger()


def run_monitors(model, monitors):
    for m in monitors:
        m.on_iter(model)

def rerun_monitors(args):
    from learning.utils.datalog import dlog, StoreToH5, TextPrinter

    from learning.experiment import Experiment
    from learning.monitor import MonitorLL, DLogModelParams, SampleFromP
    from learning.dataset import MNIST
    from learning.preproc import PermuteColumns

    from learning.isws import ISStack
    from learning.sbn  import SBN, SBNTop
    from learning.darn import DARN, DARNTop
    from learning.nade import NADE, NADETop

    import h5py


    logger.debug("Arguments %s" % args)
    tags = []


    # Layer models
    layer_models = {
        "sbn" : (SBN, SBNTop),
        "darn": (DARN, DARNTop), 
        "nade": (NADE, NADETop),
    }

    if not args.p_model in layer_models:
        raise "Unknown P-layer model %s" % args.p_model
    p_layer, p_top = layer_models[args.p_model]

    if not args.q_model in layer_models:
        raise "Unknown P-layer model %s" % args.p_model
    q_layer, q_top = layer_models[args.q_model]

    # Layer sizes
    layer_sizes = [int(s) for s in args.layer_sizes.split(",")]

    n_X = 28*28

    p_layers = []
    q_layers = []

    for ls in layer_sizes:
        n_Y = ls
        p_layers.append(
            p_layer(n_X=n_X, n_Y=n_Y, clamp_sigmoid=True)
        )
        q_layers.append(
            q_layer(n_X=n_Y, n_Y=n_X)
        )
        n_X = n_Y
    p_layers.append( p_top(n_X=n_X, clamp_sigmoid=True) )
            

    model = ISStack(
        p_layers=p_layers,
        q_layers=q_layers
    )
    model.setup()

    # Dataset
    if args.no_shuffle:
        preproc = []
        tags += ["noshuffle"]
    else:
        np.random.seed(23)
        preproc = [PermuteColumns()]

    tags.sort()

    expname = args.cont
    if expname[-1] == "/":
        expname = expname[:-1]
    

    result_dir = "reruns/%s" % os.path.basename(expname)
    results_fname = result_dir+"/results.h5"
    logger.info("Output logging to %s" % result_dir)
    os.makedirs(result_dir)
    dlog.set_handler("*", StoreToH5, results_fname)

    fname = args.cont + "/results.h5" 
    logger.info("Loading from %s" % fname)
    with h5py.File(fname, "r") as h5:
        LL = h5['learning.monitor.100.LL'][:-1]
        
        print("Final on validation:   %5.2f  (iteration %d)" % (LL[-1], LL.shape[0]))
        print("Best on validation:    %5.2f  (iteration %d)" % (np.max(LL), np.argmax(LL)))

        logger.info("Loading dataset...")
        testset = MNIST(fname="mnist_salakhutdinov.pkl.gz", which_set='test', preproc=preproc, n_datapoints=10000)

        logger.info("Setting up monitors...")
        monitors = [MonitorLL(data=testset, n_samples=[1, 5, 10, 25, 100, 500, 1000])]
        #monitors = [MonitorLL(data=testset, n_samples=[500,])]
        #monitors = [SampleFromP(n_samples=200)]

        for m in monitors:
            m.on_init(model)

        logger.info("Loading model (row %d)..." % -1)
        logger.info("LL on validation set: %f5.2" % LL[-1])
        model.model_params_from_h5(h5, row=-1)
        run_monitors(model, monitors)

        best = np.argsort(LL)[:-5:-1]
        for row in best:
            logger.info("Loading model (row %d)..." % row)
            logger.info("LL on validation set: %f5.2" % LL[row])
            model.model_params_from_h5(h5, row=row)
            run_monitors(model, monitors)
        
    logger.info("Finished.")

    #experiment.print_summary()

#=============================================================================
if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--no-shuffle', action='store_true', default=False)
    parser.add_argument('cont', 
        help="Continue a previous in result_dir")
    parser.add_argument('p_model', default="SBN", 
        help="SBN, DARN or NADE (default: SBN")
    parser.add_argument('q_model', default="SBN",
        help="SBN, DARN or NADE (default: SBN")
    parser.add_argument('layer_sizes', default="200,200,10", 
        help="Comma seperated list of sizes. Layer cosest to the data comes first")
    args = parser.parse_args()

    FORMAT = '[%(asctime)s] %(module)-15s %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

    rerun_monitors(args)
