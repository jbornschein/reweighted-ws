
import numpy as np

from learning.dataset import BarsData, FromModel, MNIST
from learning.stbp_layers import  STBPStack, SigmoidBeliefLayer, FactoizedBernoulliTop
from learning.training import BatchedSGD
from learning.train_stbp import TrainSTBP
from learning.termination import LogLikelihoodIncrease

n_vis = 28*28
n_hid = 200
n_qhid = 2*n_hid
dataset = MNIST(which_set='train')

layers=[
    SigmoidBeliefLayer( 
        unroll_scan=1,
        n_lower=n_vis,
        n_qhid=n_qhid,
    ),
    SigmoidBeliefLayer( 
        unroll_scan=1,
        n_lower=n_hid,
        n_qhid=n_qhid,
    ),
    FactoizedBernoulliTop(
        n_lower=100,
    )
]

model = STBPStack(
    layers=layers
)

#model.set_model_param('P_a', 2./n_hid*np.ones(n_hid))
#model.set_model_param('P_b', -2*np.ones(n_vis))
#model.set_model_param('P_W', W_bars)


trainer_params = {
    "n_samples"     : 25,
    "learning_rate" : 1e-3,
    "layer_discount": 0.25,
    "batch_size"    : 5,
    "recalc_LL"     : [1, 5, 25, 100]
}
trainer = TrainSTBP(**trainer_params)

termination_param = {
    "min_increase": 0.005,
    "min_epochs": 10,
    "max_epochs": 200.
}
termination = LogLikelihoodIncrease(**termination_param)

