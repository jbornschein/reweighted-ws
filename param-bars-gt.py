
import numpy as np

from learning.dataset import MNIST, ToyData, BarsData
from learning.isb import  ISB
from learning.training import BatchedSGD
from learning.train_isb import TrainISB
from learning.termination import LogLikelihoodIncrease


D = 4
n_vis = D**2
n_hid = 10
n_qhid = 2*n_hid


#----------------------------------------------------------------------
dataset = BarsData(which_set='train', n_datapoints=1000, D=D)

#----------------------------------------------------------------------
W_bars = np.zeros( (n_hid, D, D) )
for d in xrange(D):
    W_bars[  d,:,d] = 4.
    W_bars[D+d,d,:] = 4.
W_bars = W_bars.reshape((n_hid, n_vis))

model_params = {
    "unroll_scan": 1,
    "n_samples": 100,
    "n_vis": n_vis,
    "n_hid": n_hid, 
    "n_qhid": n_qhid,
}
model = ISB(**model_params)

#model.set_model_param('P_a', 2./n_hid*np.ones(n_hid))
#model.set_model_param('P_b', -2*np.ones(n_vis))
#model.set_model_param('P_W', W_bars)

#----------------------------------------------------------------------
trainer_params = {
    "learning_rate" : 1e-2,
    "batch_size"    : 1,
    "recalc_LL"     : [25, 100, 500, 'exact']

}
trainer = TrainISB(**trainer_params)

#----------------------------------------------------------------------
termination_param = {
    "min_increase": 0.005, 
    "min_epochs":  20,
    "max_epochs": 500
}
termination = LogLikelihoodIncrease(**termination_param)

