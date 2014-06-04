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
    parser.add_argument('--shuffle', default=False, action="store_true")
    parser.add_argument('out_dir', nargs='+')
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=level)


    if args.shuffle:
        np.random.seed(23)
        perm = np.random.permutation(28*28)
        perm_inv = np.argsort(perm)
    else:
        perm = np.arange(28*28)
        perm_inv = perm

    for out_dir in args.out_dir:
        fname = out_dir+"/results.h5"
        try:
            with h5py.File(fname, "r") as h5:
                print "==== %s ====" % out_dir

                logger.debug("Keys:")
                for k, v in h5.iteritems():
                    logger.debug("  %-30s   %s" % (k, v.shape))

                
                samples = h5['learning.monitor.L0.samples_prob'][-1,:,:]
                log_p = h5['learning.monitor.log_p'][-1,:]    
                samples = samples[:, perm_inv]
                #idx = np.argsort(log_p)[::-1]
                #samples = samples[idx]

                pylab.figure()
                for i in xrange(100):
                    pylab.subplot(10, 10, i+1)
                    pylab.imshow( samples[i,:].reshape((28,28)), interpolation='nearest')
                    pylab.gray()
                    pylab.axis('off')


                #W0 = h5['learning.monitor.L0.P.W'][-1,:,:]
                #pylab.figure()
                #for i in xrange(100):
                #    pylab.subplot(10, 10, i+1)
                #    print "a"
                #    pylab.imshow( W0[i,:].reshape((28,28)) )
                #    pylab.axis('off')
 
                    
                #if 'learning.monitor.10.LL' in h5:
                #    LL10 = h5['learning.monitor.10.LL'][:]
                #    print "Final LL [ 10 samples]: %.2f" % LL10[-1]
                     
        except KeyError, e:
            logger.info("Failed to read data from %s" % fname)

        except IOError, e:
            logger.info("Failed to open %s fname: %s" % (fname, e))

    pylab.legend(loc="lower right")
    pylab.show()

