
from __future__ import division

import sys
sys.path.append("../lib")

import logging
import time
import cPickle as pickle
import os
import os.path 
import errno
from shutil import copyfile

import numpy as np

import theano
import theano.tensor as T

from utils.datalog import dlog, StoreToH5, TextPrinter

from dataset import DataSet
from model import Model
from training import TrainerBase
from termination import Termination
from monitor import DLogModelParams

_logger = logging.getLogger()

class Experiment(object):
    @classmethod
    def from_param_file(cls, fname):
        experiment = cls()
        experiment.load_param_file(fname)
        return experiment

    #-------------------------------------------------------------------------

    def __init__(self):
        self.params = {}
        self.param_fname = None
        self.out_dir = None
        self.logger = _logger
        
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

        logger_fname = os.path.join(self.out_dir, "logfile.txt")
        fh = logging.FileHandler(logger_fname)
        logging

    def print_summary(self):
        logger = self.logger
        
        logger.info("Parameter file:   %s" % self.param_fname)
        logger.info("Output directory: %s" % self.out_dir)
        logger.info("-- Trainer hyperparameter --")
        for k, v in self.trainer.get_hyper_params().iteritems():
            if not isinstance(v, (int, float)):
                continue
            logger.info("  %20s: %s" % (k, v))
        #logger.info("-- Model hyperparameter --")
        #model = self.trainer.model
        #n_layers = len(model.p_layers)
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

#-----------------------------------------------------------------------------


