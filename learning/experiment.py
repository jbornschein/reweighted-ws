
from __future__ import division

import sys
sys.path.append("../lib")

import logging
import time
import cPickle as pickle
import os
import os.path 
import errno
from six import iteritems
from shutil import copyfile

import numpy as np
import h5py

import theano
import theano.tensor as T

from utils.datalog import dlog, StoreToH5, TextPrinter

from dataset import DataSet
from model import Model
from training import TrainerBase
from termination import Termination
from monitor import DLogModelParams

class Experiment(object):
    @classmethod
    def from_param_file(cls, fname):
        experiment = cls()
        experiment.load_param_file(fname)
        return experiment

    @classmethod
    def from_results(cls, path, row=-1):
        param_fname = path + "/paramfile.py"
        results_fname = path + "/results.h5"

        experiment = cls()
        experiment.load_param_file(param_fname)
        
        model = experiment.params['model']
        with h5py.File(results_fname, "r") as h5:
            model.model_params_from_h5(h5, row, basekey="mode.")

        return experiment

    #-------------------------------------------------------------------------

    def __init__(self):
        self.params = {}
        self.param_fname = None
        self.out_dir = None
        self.logger = logging.getLogger("experiment")
        
    def load_param_file(self, fname):
        self.param_fname = fname
        execfile(fname, self.params)

        self.set_trainer(self.params['trainer'])

    def setup_output_dir(self, exp_name=None, with_suffix=True):
        if exp_name is None:
            # Determine experiment name
            if self.param_fname:
                exp_name = self.param_fname
            else:
                exp_name = sys.argv[0]
 
        if with_suffix:
            # Determine suffix
            if 'PBS_JOBID' in os.environ:
                job_no = os.environ['PBS_JOBID'].split('.')[0]   # Job Number
                suffix = "j"+job_no
            elif 'SLURM_JOBID' in os.environ:
                job_no = os.environ['SLURM_JOBID']
                suffix = "j"+job_no
            else:
                suffix = time.strftime("%Y-%m-%d-%H-%M")
    
            if not with_suffix:
                suffix = "-"
    
            suffix_counter = 0
            dirname = "output/%s.%s" % (exp_name, suffix)
            while True:
                try:
                   os.makedirs(dirname)
                except OSError, e:
                    if e.errno != errno.EEXIST:
                        raise e
                    suffix_counter += 1
                    dirname = "output/%s.%s+%d" % (exp_name, suffix, suffix_counter)
                else:
                    break
        else:
            dirname = "output/%s" % (exp_name)
            try:
                os.makedirs(dirname)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise e
                
        out_dir = dirname+"/"
        self.out_dir = out_dir

        if self.param_fname:
            copyfile(self.param_fname, os.path.join(self.out_dir, "paramfile.py"))
        
    def setup_logging(self):
        assert self.out_dir

        results_fname = os.path.join(self.out_dir, "results.h5")
        dlog.set_handler("*", StoreToH5, results_fname)

        #FORMAT = '[%(asctime)s] %(module)-15s %(message)s'
        FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
        DATEFMT = "%H:%M:%S"

        formatter = logging.Formatter(FORMAT, DATEFMT)

        logger_fname = os.path.join(self.out_dir, "logfile.txt")
        fh = logging.FileHandler(logger_fname)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        root_logger = logging.getLogger("")
        root_logger.addHandler(fh)

    def print_summary(self):
        logger = self.logger
        
        logger.info("Parameter file:   %s" % self.param_fname)
        logger.info("Output directory: %s" % self.out_dir)
        logger.info("-- Trainer hyperparameter --")
        for k, v in iteritems(self.trainer.get_hyper_params()):
            if not isinstance(v, (int, float)):
                continue
            logger.info("  %20s: %s" % (k, v))
        logger.info("-- Model hyperparameter --")
        model = self.trainer.model


        desc = [str(layer.n_X) for layer in model.p_layers]
        logger.info(" %20s: %s" % ("layer sizes", "-".join(desc)))
        desc = [str(layer.__class__) for layer in model.p_layers]
        logger.info(" %20s: %s" % ("p-layers", " - ".join(desc)))
        desc = [str(layer.__class__) for layer in model.q_layers]
        logger.info(" %20s: %s" % ("q-layers", " - ".join(desc)))
    
            
        #for pl, ql in zip(model.p_layers[:-1], model.q_layers):
        #    logger.info("    %s" % l.__class__)
        #    for k, v in l.get_hyper_params().iteritems():
        #        logger.info("      %20s: %s" % (k, v))
        #logger.info("Total runtime:    %f4.1 h" % runtime)
 
    def run_experiment(self):
        self.sanity_check()

        self.trainer.load_data()
        self.trainer.compile()

        self.trainer.perform_learning()

    def continue_experiment(self, results_h5, row=-1, keep_orig_data=True):
        logger = self.logger
        self.sanity_check()

        # Never copy these keys from original .h5
        skip_orig_keys = (
            "trainer.psleep_L",
            "trainer.pstep_L"
        )

        logger.info("Copying results from %s" % results_h5)
        with h5py.File(results_h5, "r") as h5:
            if keep_orig_data:
                for key in h5.keys():
                    if key in skip_orig_keys:
                        continue
                    n_rows = h5[key].shape[0]
                    if row > -1:
                        n_rows = min(n_rows, row)
                    for r in xrange(n_rows):
                        dlog.append("orig."+key, h5[key][r])

            # Identify last row without NaN's
            #LL100 = h5['learning.monitor.100.LL']
            #row = max(np.where(np.isfinite(LL100))[0])-1
            logger.info("Continuing from row %d" % row)

            self.trainer.load_data()
            self.trainer.compile()
            self.trainer.model.model_params_from_h5(h5, row=row)

        self.trainer.perform_learning()

    #---------------------------------------------------------------
    def sanity_check(self):
        if not isinstance(self.trainer, TrainerBase):
            raise ValueError("Trainer not set properly")
        
        if not any( [isinstance(m, DLogModelParams) for m in self.trainer.epoch_monitors] ):
            self.logger.warn("DLogModelParams is not setup as an epoch_monitor. Model parameters wouldn't be saved. Adding default DLogModelParams()")
            self.trainer.epoch_monitors += DLogModelParams()

    def set_trainer(self, trainer):
        assert isinstance(trainer, TrainerBase)
        self.trainer = trainer

