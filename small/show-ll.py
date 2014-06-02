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
    parser.add_argument('out_dir', nargs='+')
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=level)

    #pylab.figsize(12, 8)
    pylab.xlabel("Epochs")
    pylab.ylabel("avg_{x~testdata} log( E_{h~q}[p(x,h)/q(h|x)]")
    pylab.ylim([-30, -10])

    for out_dir in args.out_dir:
        fname = out_dir+"/results.h5"

        try:
            with h5py.File(fname, "r") as h5:

                print "==== %s ====" % out_dir
                logger.debug("Keys:")
                for k, v in h5.iteritems():
                    logger.debug("  %-30s   %s" % (k, v.shape))
                n_steps = h5['learning.training.timing.epoch'].shape[0]

                LL100 = h5['learning.monitor.100.LL'][:]

                stepping = 1
                if LL100.shape[0] > 1.5*n_steps:
                    stepping = 2

                pylab.plot(LL100[::stepping], label=out_dir[-20:])
                
    

                if 'learning.monitor.10.LL' in h5:
                    LL10 = h5['learning.monitor.10.LL'][:]
                    print "Final LL [ 10 samples]: %.2f" % LL10[-1]

                print "Final LL [100 samples]: %.2f" % LL100[-1]

                if 'learning.monitor.500.LL' in h5:
                    LL500 = h5['learning.monitor.500.LL'][:]
                    print "Final LL [500 samples]: %.2f" % LL500[-1]
                     
        except KeyError, e:
            logger.info("Failed to read data from %s" % fname)

        except IOError, e:
            logger.info("Failed to open %s fname: %s" % (fname, e))

    pylab.legend(loc="lower right")
    pylab.show()

