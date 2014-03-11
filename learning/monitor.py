#!/usr/bin/env python 

from __future__ import division

import abc
import logging

import numpy as np

import theano 
import theano.tensor as T

from dataset import DataSet
from model import Model
from hyperbase import HyperBase
import utils.datalog as datalog

_logger = logging.getLogger(__name__)

class Monitor(HyperBase):
    """ Abtract base class to monitor stuff """

    __metaclass__ = abc.ABCMeta

    def __init__(self):   
        """ 
        """
        self.dlog = datalog.getLogger(__name__)
        self.logger = logging.getLogger(__name__)

    def compile(self):
        pass

    def on_init(self, model):
        """ Called after the model has been initialized; directly before 
            the first learning epoch will be performed 
        """
        self.on_iter(model)

    @abc.abstractmethod
    def on_iter(self, model):
        """ Called whenever a full training epoch has been performed
        """
        pass

class DLogHyperParams(Monitor):
    def __init__(self):
        super(DLogHyperParams, self).__init__()

    def on_iter(self, model):
        model.model_params_to_dlog(self.dlog)

class DLogModelParams(Monitor):
    """
    Write all model parameters to a DataLogger called "model_params".
    """
    def __init__(self):
        super(DLogModelParams, self).__init__()

    def on_iter(self, model):
        self.logger.info("Saving model parameters")
        model.model_params_to_dlog(self.dlog)


class MonitorAnalyse(Monitor):
    def __init__(self, data, n_samples):
        super(MonitorAnalyse, self).__init__()

        assert isinstance(data, DataSet)
        self.data = data

        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        data = self.data
        assert isinstance(data, DataSet)
        self.train_X = theano.shared(data.X, "train_X")
        self.train_Y = theano.shared(data.Y, "train_Y")


        self.logger.info("compiling do_analyze")
        n_samples = T.iscalar("n_samples")
        batch_idx  = T.iscalar('batch_idx')
        batch_size = T.iscalar('batch_size')

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[first:last]
        
        log_PX, w, log_p, log_q, KL, Hp, Hq = model.log_likelihood(X_batch, n_samples=n_samples)

        # calculate
        #log_KLpq()
        #Hp = [.,.,.]
        #Hq = [.,.,.]

        self.do_analyze = theano.function(  
                            inputs=[batch_idx, batch_size, n_samples], 
                            outputs=[batch_log_PX, log_KL, Hp, Hq], 
                            name="do_analyze",
                            allow_input_downcast=True,
                            on_unused_input='warn')

    def on_init(self, model):
        self.compile(model)
        self.on_iter(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_datapoints = self.data.n_datapoints

        #
        for K in n_samples:
            if K <= 10:
                batch_size = 100
            elif K <= 100:
                batch_size = 10
            else:
                batch_size = 1
    

            # Iterate over dataset
            L = 0
            for batch_idx in xrange(n_datapoints//batch_size):
                log_PX = self.do_loglikelihood(batch_idx, batch_size, K)
                L += log_PX
            L /= n_datapoints

            self.logger.info("Test LL for %d datapoints with %s samples: %5.2f" % (n_datapoints, K, L))
            self.dlog.append("LL_%d"%K, L)
 

class MonitorLL(Monitor):
    """ Monitor the LL after each training epoch on an arbitrary 
        test or validation data set
    """
    def __init__(self, data, n_samples):
        super(MonitorLL, self).__init__()

        assert isinstance(data, DataSet)
        self.data = data

        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        data = self.data
        assert isinstance(data, DataSet)
        self.train_X = theano.shared(data.X, "train_X")
        self.train_Y = theano.shared(data.Y, "train_Y")

        batch_idx  = T.iscalar('batch_idx')
        batch_size = T.iscalar('batch_size')

        self.logger.info("compiling do_loglikelihood")
        n_samples = T.iscalar("n_samples")
        batch_idx = T.iscalar("batch_idx")
        batch_size = T.iscalar("batch_size")

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[first:last]
        
        log_PX, _, _, _, KL, Hp, Hq = model.log_likelihood(X_batch, n_samples=n_samples)
        batch_log_PX = T.sum(log_PX)
        batch_KL = [T.sum(kl) for kl in KL]
        batch_Hp = [T.sum(hp) for hp in Hp]
        batch_Hq = [T.sum(hq) for hq in Hq]

        self.do_loglikelihood = theano.function(  
                            inputs=[batch_idx, batch_size, n_samples], 
                            outputs=[batch_log_PX] + batch_KL + batch_Hp + batch_Hq, 
                            name="do_likelihood",
                            allow_input_downcast=True,
                            on_unused_input='warn')

    def on_init(self, model):
        self.compile(model)
        self.on_iter(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_datapoints = self.data.n_datapoints

        #
        for K in n_samples:
            if K <= 10:
                batch_size = 100
            elif K <= 100:
                batch_size = 10
            else:
                batch_size = 1
    
            n_layers = len(model.layers)

            L = 0
            KL = np.zeros(n_layers)
            Hp = np.zeros(n_layers)
            Hq = np.zeros(n_layers)
        
            # Iterate over dataset
            for batch_idx in xrange(n_datapoints//batch_size):
                outputs = self.do_loglikelihood(batch_idx, batch_size, K)
                batch_L , outputs = outputs[0], outputs[1:]
                batch_KL, outputs = outputs[:n_layers], outputs[n_layers:]
                batch_Hp, outputs = outputs[:n_layers], outputs[n_layers:]
                batch_Hq          = outputs[:n_layers]
                
                L += batch_L
                KL += np.array(batch_KL)
                Hp += np.array(Hp)
                Hq += np.array(Hq)
                
            L /= n_datapoints
            KL /= n_datapoints
            Hp /= n_datapoints
            Hq /= n_datapoints

            prefix = "%d." % K

            self.logger.info("MonitorLL (%d datpoints, %d samples): LL=%5.2f KL=%s" % (n_datapoints, K, L, KL))
            self.dlog.append_all({
                prefix+"LL": L,
                prefix+"KL": KL,
                prefix+"Hp": Hp,
                prefix+"Hq": Hq,
            })
        
class SampleFromP(Monitor):
    """ Draw a number of samples from the P-Model """
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def on_iter(self, model):
        raise NotImplemented()

