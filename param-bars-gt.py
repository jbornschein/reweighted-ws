
from __future__ import division

import numpy as np

from learning.dataset import MNIST, ToyData, BarsData, FromModel
from learning.isb import  ISB
from learning.training import BatchedSGD
from learning.train_isb import TrainISB
from learning.termination import LogLikelihoodIncrease


n_datapoints = 1000
D = 5
n_vis = D**2
n_hid = 2*D

W_bars = np.zeros( (n_hid, D, D) )
for d in xrange(D):
    W_bars[  d,:,d] = 20.
    W_bars[D+d,d,:] = 20.
W_bars = W_bars.reshape((n_hid, n_vis))
P_a = -np.log(n_hid/2.-1)*np.ones(n_hid)
P_b = -10*np.ones(n_vis)

gt_model = ISB(n_vis=n_vis, n_hid=n_hid)
gt_model.set_model_param('P_a', P_a)
gt_model.set_model_param('P_b', P_b)
gt_model.set_model_param('P_W', W_bars)

dataset = FromModel(gt_model, n_datapoints=n_datapoints)

#----------------------------------------------------------------------
n_hid  = 15
n_qhid = 20
model_params = {
    "unroll_scan": 1,
    "n_samples": 25,
    "n_vis": n_vis,
    "n_hid": n_hid,
    "n_qhid": 20,
}

model = ISB(**model_params)
#model.set_model_param('P_a', P_a)
#model.set_model_param('P_b', P_b)
#model.set_model_param('P_W', W_bars)

#----------------------------------------------------------------------
trainer_params = {
    "learning_rate" : 1e-2,
    "batch_size"    : 1,
    "recalc_LL"     : [10, 25, 100, 500] #, 'exact']
}
trainer = TrainISB(**trainer_params)

#----------------------------------------------------------------------
termination_param = {
    "min_increase": 0.005, 
    "min_epochs":  10,
    "max_epochs": 500
}
termination = LogLikelihoodIncrease(**termination_param)

