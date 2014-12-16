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
    parser.add_argument('--param', default="model.L0.P.W")
    parser.add_argument('--transpose', '-T', action="store_true", default=False)
    parser.add_argument('--shape', default="28,28",
            help="Shape for each samples (default: 28,28)")
    parser.add_argument('--row', default=-1, type=int,
            help="Iteration to visualize")
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
    param = args.param
    try:
        with h5py.File(fname, "r") as h5:

            logger.debug("Keys:")
            for k, v in h5.iteritems():
                logger.debug("  %-30s   %s" % (k, v.shape))
                
            row = args.row
            total_rows = h5[param].shape[0]
            logger.info("Visualizing row %d of %d..." % (args.row, total_rows))

            W0 = h5[param][row,:,:]
            if args.transpose:
                W0 = W0.T
            H, D = W0.shape

            if 'preproc.permute_columns.permutation_inv' in h5:
                logger.debug("Experiment used PermuteColumns preproc -- loading inv_perm")
                perm_inv = h5['preproc.permute_columns.permutation_inv'][:]
            else:
                perm_inv = np.arange(D)

    except KeyError, e:
        logger.info("Failed to read data from %s: %s" % (fname, e))
        exit(1)

    except IOError, e:
        logger.info("Failed to open %s: %s" % (fname, e))
        exit(1)

    shape = tuple([int(s) for s in args.shape.split(",")])

    width = int(np.sqrt(H))
    height = width
    if width*height < H:
        width = width + 1
    if width*height < H:
        height = height + 1
        
    logger.debug("Using shape: %s -- %s" % (args.shape, shape))
    assert len(shape) == 2

    pylab.figure()
    for h in xrange(H):
        pylab.subplot(width, height, h+1)
        pylab.imshow( W0[h,perm_inv].reshape(shape), interpolation='nearest')
        pylab.gray()
        pylab.axis('off')

    pylab.legend(loc="lower right")
    pylab.show(block=True)

