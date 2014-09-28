#!/usr/bin/env python 

from __future__ import division

import sys

import logging
from time import time
import cPickle as pickle

import numpy as np
import h5py


import pylab
#import theano
#import theano.tensor as T

_logger = logging.getLogger()

#=============================================================================
if __name__ == "__main__":
    import argparse 

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action="store_true", default=False)
    parser.add_argument('--dataset', '-d', default="valiset")
    parser.add_argument('--samples', '-s', default=100)
    parser.add_argument('--stacked', action="store_true", default=False)
    parser.add_argument('out_dir', nargs=1)
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=level)

    fname = args.out_dir[0]+"/results.h5"
    table = "%s.spl%d.Hp" % (args.dataset, args.samples)

    try:
        with h5py.File(fname, "r") as h5:
            Hp = h5[table][:]
                
    except KeyError, e:
        logger.info("Failed to read data from %s: %s" % (fname, e))
        exit(1)

    except IOError, e:
        logger.info("Failed to open %s fname: %s" % (fname, e))
        exit(1)

    epochs = Hp.shape[0]
    n_layers = Hp.shape[1]

    if args.stacked:
        ylim = 2*Hp[-1].sum()
        pylab.ylim([ylim, 0])
        pylab.stackplot(np.arange(epochs), Hp[:,::-1].T)
    else:
        ylim = 2*Hp[-1].min()
        pylab.ylim([ylim, 0])
        pylab.plot(Hp)

    #pylab.figsize(12, 8)
    pylab.xlabel("Epochs")
    #pylab.ylabel("avg_{x~testdata} log( E_{h~q}[p(x,h)/q(h|x)]")
    pylab.legend(["layer %d" % i for i in xrange(n_layers)], loc="lower right")
    pylab.show(block=True)

