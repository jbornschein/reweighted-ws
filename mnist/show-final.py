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

    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action="store_true", default=False)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('results', nargs='+')
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=level)

    best_dir = ""
    best_ll = -np.inf

    for result_dir in args.results:
        fname = result_dir+"/results.h5"

        try:
            log.debug("opening %s" % result_dir)
            with h5py.File(fname, "r") as h5:

                log.debug("Keys:")
                for k, v in h5.iteritems():
                    log.debug("  %-30s   %s" % (k, v.shape))
                n_steps = h5['learning.training.timing.epoch'].shape[0]

                path = "learning.monitor.%d.LL" % args.samples
                final_ll = h5[path][-1]

                print "%-50s: %f" % (result_dir, final_ll)

                if final_ll > best_ll:
                    best_ll = final_ll
                    best_dir = result_dir
                #print "Final LL [100 samples]: %.2f" % LL100[-1]

                #if 'learning.monitor.500.LL' in h5:
                #    LL500 = h5['learning.monitor.500.LL'][:]
                #    print "Final LL [500 samples]: %.2f" % LL500[-1]
                     
        except KeyError, e:
            log.info("Failed to read data from %s" % fname)

        except IOError, e:
            log.info("Failed to open %s fname: %s" % (fname, e))


    print "="*70
    print "%-50s: %f" % (best_dir, best_ll)
