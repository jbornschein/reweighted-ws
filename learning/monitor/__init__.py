#!/usr/bin/env python 

from __future__ import division

import abc
import logging

import numpy as np

import theano 
import theano.tensor as T

from learning.dataset import DataSet
from learning.model import Model
from learning.hyperbase import HyperBase
import learning.utils.datalog as datalog

_logger = logging.getLogger("learning.monitor")

#-----------------------------------------------------------------------------
class Monitor(HyperBase):
    """ Abtract base class to monitor stuff """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):   
        """ 

        Parameters
        ----------
        name:   str
            dlog channel to use
        """
        if name is None:
            name = self.__class__.__name__

        self.dlog = datalog.getLogger(name)
        self.logger = logging.getLogger(name)

    def compile(self):
        pass

    def on_init(self, model):
        """ Called after the model has been initialized; directly before 
            the first learning epoch will be performed 
        """
        pass

    @abc.abstractmethod
    def on_iter(self, model):
        """ Called whenever a full training epoch has been performed
        """
        pass


#-----------------------------------------------------------------------------
class DLogHyperParams(Monitor):
    def __init__(self, name=None):
        if name is None:
            name="hyper"
        super(DLogHyperParams, self).__init__(name)

    def on_iter(self, model, name=None):
        model.hyper_params_to_dlog(self.dlog)


#-----------------------------------------------------------------------------
class DLogModelParams(Monitor):
    """
    Write all model parameters to a DataLogger called "model_params".
    """
    def __init__(self, name=None):
        if name is None:
            name="model"
        super(DLogModelParams, self).__init__(name)

    def on_iter(self, model):
        self.logger.info("Saving model parameters")
        model.model_params_to_dlog(self.dlog)


#-----------------------------------------------------------------------------
class MonitorLL(Monitor):
    """ Monitor the LL after each training epoch on an arbitrary 
        test or validation data set
    """
    def __init__(self, data, n_samples, name=None):
        super(MonitorLL, self).__init__(name)

        assert isinstance(data, DataSet)
        self.dataset = data

        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        dataset = self.dataset
        X, Y = dataset.preproc(dataset.X, dataset.Y)
        self.X = theano.shared(X, "X")
        self.Y = theano.shared(Y, "Y")

        self.logger.info("compiling do_loglikelihood")
        n_samples = T.iscalar("n_samples")
        batch_idx = T.iscalar("batch_idx")
        batch_size = T.iscalar("batch_size")

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch, Y_batch = dataset.late_preproc(self.X[first:last], self.Y[first:last])
        
        log_PX, _, _, _, KL, Hp, Hq = model.log_likelihood(X_batch, n_samples=n_samples)
        batch_L  = T.sum(log_PX)
        batch_L2 = T.sum(log_PX**2)
        batch_KL = [T.sum(kl) for kl in KL]
        batch_Hp = [T.sum(hp) for hp in Hp]
        batch_Hq = [T.sum(hq) for hq in Hq]

        self.do_loglikelihood = theano.function(  
                            inputs=[batch_idx, batch_size, n_samples], 
                            outputs=[batch_L, batch_L2] + batch_KL + batch_Hp + batch_Hq, 
                            name="do_likelihood")

    def on_init(self, model):
        self.compile(model)

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
    
            n_layers = len(model.p_layers)

            L  = 0
            L2 = 0
            KL = np.zeros(n_layers)
            Hp = np.zeros(n_layers)
            Hq = np.zeros(n_layers)
        
            # Iterate over dataset
            for batch_idx in xrange(n_datapoints//batch_size):
                outputs = self.do_loglikelihood(batch_idx, batch_size, K)
                batch_L , outputs = outputs[0], outputs[1:]
                batch_L2, outputs = outputs[0], outputs[1:]
                batch_KL, outputs = outputs[:n_layers], outputs[n_layers:]
                batch_Hp, outputs = outputs[:n_layers], outputs[n_layers:]
                batch_Hq          = outputs[:n_layers]
                
                L  += batch_L
                L2 += batch_L2
                KL += np.array(batch_KL)
                Hp += np.array(batch_Hp)
                Hq += np.array(batch_Hq)
                
            L_se  = np.sqrt((L2 - (L*L)/n_datapoints) / (n_datapoints - 1)) 
            L_se *= 1.96 / np.sqrt(n_datapoints)

            L  /= n_datapoints
            KL /= n_datapoints
            Hp /= n_datapoints
            Hq /= n_datapoints

            global validation_LL
            validation_LL = L

            self.logger.info("(%d datpoints, %d samples): LL=%5.2f +-%3.2f; Hp=%s" % (n_datapoints, K, L, L_se, Hp))

            prefix = "spl%d." % K
            self.dlog.append_all({
                prefix+"LL": L,
                prefix+"KL": KL,
                prefix+"Hp": Hp,
                prefix+"Hq": Hq,
            })
        

#-----------------------------------------------------------------------------
class SampleFromP(Monitor):
    """ Draw a number of samples from the P-Model """
    def __init__(self, n_samples=100, data=None):
        super(SampleFromP, self).__init__()

        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)

        self.logger.info("compiling do_sample")

        n_samples = T.iscalar('n_samples')
        n_samples.tag.test_value = self.n_samples
        samples, log_p = model.sample_p(n_samples)

        try:
            expected_samples = [model.p_layers[0].sample_expected(samples[1])]
            self.support_sample_expected = True
        except:
            expected_samples = []
            self.support_sample_expected = False

        self.do_sample = theano.function(
                            inputs=[n_samples],
                            outputs=[log_p] + samples + expected_samples,
                            name="do_sample")

    def on_init(self, model):
        self.compile(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_layers = len(model.p_layers)

        self.logger.info("SampleFromP(n_samples=%d)" % n_samples)

        outputs = self.do_sample(n_samples)
        log_p, outputs   = outputs[0], outputs[1:]
        samples, outputs = outputs[0:n_layers], outputs[n_layers:]

        self.dlog.append("log_p", log_p)
        for l in xrange(n_layers):
            prefix = "L%d" % l
            self.dlog.append(prefix, samples[l])

        if self.support_sample_expected:
            expected_samples = outputs[0]
            self.dlog.append("L0_expected", expected_samples)
            
