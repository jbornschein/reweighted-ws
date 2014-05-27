#
# Wake-sleep experiment with the same parameters as in
#
#   Neural Variational Inference and Learning in Belief Networks (Andriy Mnih, Karol Gregor; 2014)
#   http://arxiv.org/abs/1402.0030
#

import numpy as np

from learning.dataset import BarsData, FromModel, MNIST
from learning.preproc import PermuteColumns
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams, SampleFromP

from learning.isws import ISStack
from learning.sbn  import SBN, SBNTop
from learning.darn import DARN, DARNTop
from learning.nade import NADE, NADETop

n_vis = 28*28

permute = PermuteColumns()
dataset = MNIST(fname="mnist_salakhutdinov.pkl.gz", which_set='salakhutdinov_train', preproc=[permute], n_datapoints=59000)
valiset = MNIST(fname="mnist_salakhutdinov.pkl.gz", which_set='salakhutdinov_valid', preproc=[permute], n_datapoints=1000)
testset = MNIST(fname="mnist_salakhutdinov.pkl.gz", which_set='test', preproc=[permute], n_datapoints=10000)

p_layers=[
    NADE(
        n_X=n_vis,
        n_Y=200,
        clamp_sigmoid=True,
    ),
    NADETop( 
        n_X=200,
        clamp_sigmoid=True,
    ),
]

q_layers=[
    NADE(
        n_Y=n_vis,
        n_X=200,
        clamp_sigmoid=True,
    )
]

model = ISStack(
    p_layers=p_layers,
    q_layers=q_layers,
)

trainer = Trainer(
    n_samples=25,
    learning_rate_p=3e-3,
    learning_rate_q=3e-3,
    learning_rate_s=3e-3,
    layer_discount=1.0,
    batch_size=20,
    dataset=dataset, 
    model=model,
    termination=EarlyStopping(min_epochs=500, max_epochs=500),
    #step_monitors=[MonitorLL(data=smallset, n_samples=[1, 5, 25, 100])],
    epoch_monitors=[MonitorLL(data=valiset, n_samples=[100]), DLogModelParams(), SampleFromP(n_samples=100)],
    final_monitors=[MonitorLL(data=testset, n_samples=[1, 5, 10, 25, 100, 500, 1000])],
    #monitor_nth_step=100,
)
