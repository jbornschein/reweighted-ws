#!/usr/bin/env python

from __future__ import division

import sys

import logging
from time import time
import cPickle as pickle

import numpy as np

_logger = logging.getLogger()

#=============================================================================
if __name__ == "__main__":
    import argparse 

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('param_file')
    parser.add_argument('result_dir', nargs='?', default=None,
        help="Continue a previous in result_dir")
    args = parser.parse_args()

    from learning.utils.datalog import dlog, StoreToH5, TextPrinter
    from learning.experiment import Experiment

    import theano
    import theano.tensor as T

    FORMAT = '[%(asctime)s] %(module)-15s %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

    experiment = Experiment.from_param_file(args.param_file)
    experiment.setup_output_dir(args.param_file, with_suffix=(not args.overwrite))
    experiment.setup_logging()
    experiment.print_summary()

    if args.result_dir is None:
        experiment.run_experiment()
    else:
        experiment.continue_experiment(args.result_dir+"/results.h5")
    
    logger.info("Finished. Exiting")
    experiment.print_summary()
