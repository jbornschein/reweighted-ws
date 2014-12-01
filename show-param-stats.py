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
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('results', nargs='+')
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=level)

    ylim = +np.inf
    for result_dir in args.results:
        fname = result_dir+"/results.h5"

        try:
            with h5py.File(fname, "r") as h5:

                print "==== %s ====" % result_dir
                logger.debug("Keys:")
                for k, v in h5.iteritems():
                    logger.debug("  %-30s   %s" % (k, v.shape))

                for k,v in h5.iteritems():
                    if not 'model.' in k:
                        continue

                    if not args.filter is None:
                        if args.filter not in k:
                            continue
    
                    values = v[:]
                    iterations = v.shape[0]
                    values = values.reshape( [iterations, -1] )
                    
                    v_min  = np.min(values, axis=1)
                    v_max  = np.max(values, axis=1)
                    v_mean = np.mean(values, axis=1)

                    pylab.errorbar(np.arange(iterations), v_mean, yerr=[v_max-v_mean, v_mean-v_min], label=k)
                    
        except KeyError as e:
            logger.info("Failed to read data from %s: %s" % (fname, e))

        except IOError as e:
            logger.info("Failed to open %s fname: %s" % (fname, e))


    #pylab.figsize(12, 8)
    pylab.xlabel("Epochs")
    pylab.ylabel("")
    pylab.legend(loc="lower right")
    pylab.show(block=True)

