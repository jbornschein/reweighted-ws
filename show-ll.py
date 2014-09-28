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
    parser.add_argument('out_dir', nargs='+')
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=level)

    ylim = 0.
    for out_dir in args.out_dir:
        fname = out_dir+"/results.h5"

        table = "%s.spl%d.LL" % (args.dataset, args.samples)

        try:
            with h5py.File(fname, "r") as h5:

                print "==== %s ====" % out_dir
                logger.debug("Keys:")
                for k, v in h5.iteritems():
                    logger.debug("  %-30s   %s" % (k, v.shape))


                LL = h5[table][:]
                LL_final = LL[-1]
                ylim = min(ylim, 2*LL_final)

                pylab.plot(LL, label=out_dir[-20:])
                print "Final LL [%d samples]: %.2f" % (args.samples, LL_final)
                print "valiset-final [%d samples]: %.2f" % (args.samples, h5["final-valiset.spl1000.LL"][-1])
                print "testset-final [%d samples]: %.2f" % (args.samples, h5["final-testset.spl1000.LL"][-1])

                
        except KeyError, e:
            logger.info("Failed to read data from %s: %s" % (fname, e))

        except IOError, e:
            logger.info("Failed to open %s fname: %s" % (fname, e))


    #pylab.figsize(12, 8)
    pylab.ylim([ylim, 0])
    pylab.xlabel("Epochs")
    pylab.ylabel("avg_{x~testdata} log( E_{h~q}[p(x,h)/q(h|x)]")
    pylab.legend(loc="lower right")
    pylab.show(block=True)

