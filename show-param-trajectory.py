#!/usr/bin/env python2

from __future__ import division, print_function

import logging
import h5py
import numpy as np
import tsne
import pylab

#x2 = tsne.bh_sne(x)


if __name__ == "__main__":
    import sys 
    import argparse

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action="store_true", default=False)
    parser.add_argument('--param', type=str, default="L0.P.W_mu")
    parser.add_argument('result_dir', nargs='+')
    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '[%(asctime)s] %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=level)

    P_all = None
    N_iter = []
    for i, d in enumerate(args.result_dir):
        fname = d+"/results.h5"
        with h5py.File(fname, 'r') as h5:
            logger.debug("Keys:")
            for k, v in h5.iteritems():
                logger.debug("  %-30s   %s" % (k, v.shape))
  
            key = "model." + args.param
            P = h5[key][:]

            n_iter = P.shape[0]
            P = P.reshape([n_iter, -1])

            mask = np.isfinite(P).all(axis=1)
            P = P[mask]
            logger.info("%s: loaded %d iterations (%d contained NaNs)" % (d, mask.sum(), n_iter-mask.sum()))
            N_iter.append(P.shape[0])

            if P_all is None:
                P_all = P
            else:
                P_all = np.concatenate([P_all, P])

    P_all = P_all.astype(np.float)
    logger.info("Running T-SNE on %s" % str(P_all.shape))

    P2_all = tsne.bh_sne(P_all, pca_d=None, perplexity=10, theta=0.5)

    for n_iter in N_iter:
        P2 = P2_all[:n_iter]
        P2_all = P2_all[n_iter:]
    
        c = np.linspace(0, 1, n_iter)
        pylab.scatter(P2[:,0], P2[:,1], c=c)
    pylab.show(block=True)
    
