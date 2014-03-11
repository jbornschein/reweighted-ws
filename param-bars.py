
import numpy as np

from learning.dataset import BarsData, FromModel, MNIST
from learning.stbp_layers import  STBPStack, SigmoidBeliefLayer, FactoizedBernoulliTop
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams

n_vis = 5*5
n_hid = 20
n_qhid = 2*n_hid

dataset = BarsData(which_set='train', n_datapoints=1000)
valiset = BarsData(which_set='valid', n_datapoints=100)

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

trainer = Trainer(
    n_samples=10,
    learning_rate_p=1e-1,
    learning_rate_q=1e-1,
    layer_discount=0.50,
    batch_size=10,
    data=dataset, 
    model=model,
    termination=EarlyStopping(),
    epoch_monitors=[DLogModelParams()],
    step_monitors=[MonitorLL(data=valiset, n_samples=[1, 5, 25, 100])],
    monitor_nth_step=100
)

