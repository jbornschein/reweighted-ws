
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
from training import Trainer
from termination import TerminationCriterium

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
        
    def load_param_file(self, fname):
        self.param_fname = fname
        execfile(fname, self.params)

    def setup_output_dir(self, exp_name=None):
        if exp_name is None:
            # Determine experiment name
            if self.param_fname:
                exp_name = self.param_fname
            else:
                exp_name = sys.argv[0]

        # Determine suffix
        if 'PBS_JOBID' in os.environ:
            job_no = os.environ['PBS_JOBID'].split('.')[0]   # Job Number
            suffix = "d"+job_no
        elif 'SLURM_JOBID' in os.environ:
            job_no = os.environ['SLURM_JOBID']
            suffix = "d"+job_no
        else:
            suffix = time.strftime("%Y-%m-%d-%H-%M")

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
                
        out_dir = dirname+"/"
        self.out_dir = out_dir

        if self.param_fname:
            copyfile(self.param_fname, os.path.join(self.out_dir, "paramfile.py"))
        

    def setup_logging(self):
        assert self.out_dir

        results_fname = os.path.join(self.out_dir, "results.h5")
        dlog.set_handler("*", StoreToH5, results_fname)
        dlog.set_handler(["L", "L_sgd"], TextPrinter)
        
    def setup_experiment(self):
        params = self.params
    
        self.set_dataset(params['dataset'])
        self.set_model(params['model'])
        self.set_trainer(params['trainer'])
        self.set_termination(params['termination'])

    def dlog_model_params(self):
        model = self.model

        _logger.info("Saving model params to H5")
        for name, val in model.get_model_params().iteritems():
            val = val.get_value()
            dlog.append(name, val)

    def run_experiment(self):
        self.sanity_check()

        dataset = self.dataset
        model = self.model
        trainer = self.trainer
        termination = self.termination
        
        trainer.set_data(dataset)
        trainer.set_model(model)
        trainer.compile()

        self.dlog_model_params()
        termination.reset()
        continue_learning = True
        epoch = 0
        LL = []
        while continue_learning:
            L = trainer.perform_epoch()
                
            dlog.append("L", L)
            self.dlog_model_params()

            # Converged?
            continue_learning = termination.continue_learning(L)

    #---------------------------------------------------------------
    
    def sanity_check(self):
        if not isinstance(self.dataset, DataSet):
            raise ValueError("DataSet not set properly")

        if not isinstance(self.model, Model):
            raise ValueError("Model not set properly")

        if not isinstance(self.trainer, Trainer):
            raise ValueError("Trainer not set properly")

        if not isinstance(self.termination, TerminationCriterium):
            raise ValueError("Termination not set properly")

    def set_dataset(self, dataset):
        assert isinstance(dataset, DataSet)
        self.dataset = dataset

    def set_model(self, model):
        assert isinstance(model, Model)
        self.model = model

    def set_trainer(self, trainer):
        assert isinstance(trainer, Trainer)
        self.trainer = trainer

    def set_termination(self, termination):
        assert isinstance(termination, TerminationCriterium)
        self.termination = termination

#-----------------------------------------------------------------------------


