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


#-----------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------
class DLogHyperParams(Monitor):
    def __init__(self):
        super(DLogHyperParams, self).__init__()

    def on_iter(self, model):
        model.hyper_params_to_dlog(self.dlog)


#-----------------------------------------------------------------------------
class DLogModelParams(Monitor):
    """
    Write all model parameters to a DataLogger called "model_params".
    """
    def __init__(self):
        super(DLogModelParams, self).__init__()

    def on_iter(self, model):
        self.logger.info("Saving model parameters")
        model.model_params_to_dlog(self.dlog)


#-----------------------------------------------------------------------------
class MonitorLL(Monitor):
    """ Monitor the LL after each training epoch on an arbitrary 
        test or validation data set
    """
    def __init__(self, data, n_samples):
        super(MonitorLL, self).__init__()

        assert isinstance(data, DataSet)
        self.dataset = data

        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        dataset = self.dataset
        self.train_X = theano.shared(dataset.X, "train_X")
        self.train_Y = theano.shared(dataset.Y, "train_Y")

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
                            name="do_likelihood")

    def on_init(self, model):
        self.compile(model)
        self.on_iter(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_datapoints = self.dataset.n_datapoints

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

            global validation_LL
            validation_LL = L

            self.logger.info("MonitorLL (%d datpoints, %d samples): LL=%5.2f KL=%s" % (n_datapoints, K, L, KL))
            self.dlog.append_all({
                prefix+"LL": L,
                prefix+"KL": KL,
                prefix+"Hp": Hp,
                prefix+"Hq": Hq,
            })
        

#-----------------------------------------------------------------------------
class SampleFromP(Monitor):
    """ Draw a number of samples from the P-Model """
    def __init__(self, data, n_samples=100):
        super(SampleFromP, self).__init__()

        assert isinstance(data, DataSet)
        self.dataset = data
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)

        self.logger.info("compiling do_sample")

        n_samples = T.iscalar('n_samples')
        n_samples.tag.test_value = self.n_samples
        samples, log_p = model.sample_p(n_samples)

        self.do_sample = theano.function(
                            inputs=[n_samples],
                            outputs=[log_p] + samples,
                            name="do_sample")

    def on_init(self, model):
        self.logger.info("SampleFromP.on_init()")
        
        self.compile(model)
        self.on_iter(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_layers = len(model.layers)

        outputs = self.do_sample(n_samples)
        log_p = outputs[0]
        samples = outputs[1:]

        self.logger.info("SampleFromP(n_samples=%d)" % n_samples)
        self.dlog.append("log_p", log_p)
        for l in xrange(n_layers):
            prefix = "L%d" % l
            self.dlog.append(prefix, samples[l])

