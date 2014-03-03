
from __future__ import division

import numpy as np

from learning.dataset import BarsData, FromModel
from learning.stbp_layers import  STBPStack, SigmoidBeliefLayer, FactoizedBernoulliTop
from learning.training import BatchedSGD
from learning.train_stbp import TrainSTBP
from learning.termination import LogLikelihoodIncrease


n_datapoints = 1000
D = 5
n_vis = D**2
n_hid = 2*D
n_qhid = n_hid

W_bars = np.zeros( (n_hid, D, D) )
for d in xrange(D):
    W_bars[  d,:,d] = 20.
    W_bars[D+d,d,:] = 20.
W_bars = W_bars.reshape((n_hid, n_vis))
P_a = -np.log(n_hid/2.-1)*np.ones(n_hid)
P_b = -10*np.ones(n_vis)

gt_layers=[
    SigmoidBeliefLayer( 
        n_lower=n_vis,
        n_qhid=n_qhid,
    ),
    FactoizedBernoulliTop(
        n_lower=n_hid
    )
]

gt_layers[0].set_model_param('b', P_b)
gt_layers[0].set_model_param('W', W_bars)
gt_layers[1].set_model_param('a', P_a)

gt_model = STBPStack(
    layers=gt_layers
)

dataset = FromModel(gt_model, n_datapoints=n_datapoints)

#----------------------------------------------------------------------
n_qhid = 20

layers=[
    SigmoidBeliefLayer( 
        unroll_scan=1,
        n_lower=n_vis,
        n_qhid=n_qhid,
    ),
    FactoizedBernoulliTop(
        n_lower=n_hid,
    )
]

layers[0].set_model_param('b', P_b)
layers[0].set_model_param('W', W_bars)
#layers[1].set_model_param('a', P_a)

model = STBPStack(
    layers=layers
)

#----------------------------------------------------------------------
trainer_params = {
    "n_samples"     : 25,
    "learning_rate" : 1e-2,
    "layer_discount": 1.,
    "batch_size"    : 1,
    "recalc_LL"     : [1, 5, 10, 25, 100, 500] #, 'exact']
}
trainer = TrainSTBP(**trainer_params)

#----------------------------------------------------------------------
termination_param = {
    "min_increase": 0.002, 
    "min_epochs":  10,
    "max_epochs": 500
}
termination = LogLikelihoodIncrease(**termination_param)

