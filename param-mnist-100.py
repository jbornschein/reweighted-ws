
import numpy as np

from learning.dataset import BarsData, FromModel, MNIST
from learning.stbp_layers import  STBPStack, SigmoidBeliefLayer, FactoizedBernoulliTop
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease
from learning.monitor import MonitorLL, DLogModelParams

n_vis = 28*28
n_hid = 100
n_qhid = 2*n_hid

dataset = MNIST(which_set='train')
valiset = MNIST(which_set='valid', n_datapoints=1000)
smallset = MNIST(which_set='valid', n_datapoints=100)

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

model = STBPStack(
    layers=layers
)


termination = LogLikelihoodIncrease(
    min_increase=0.005,
    min_epochs=10,
    max_epochs=200.
)

trainer = Trainer(
    n_samples=10,
    learning_rate_p=1e-3,
    learning_rate_q=1e-3,
    layer_discount=0.25,
    batch_size=10,
    data=dataset, 
    model=model,
    termination=termination,
    epoch_monitors=[DLogModelParams()],
    step_monitors=[MonitorLL(data=smallset, n_samples=[1, 5, 25, 100])],
    monitor_nth_step=100,
)

