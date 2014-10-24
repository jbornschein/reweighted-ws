#

import numpy as np

from learning.dataset import BarsData, FromModel, MNIST
from learning.preproc import PermuteColumns
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams, SampleFromP

from learning.models.rws  import LayerStack
from learning.models.sbn  import SBN, SBNTop
from learning.models.darn import DARN, DARNTop
from learning.models.nade import NADE, NADETop

n_vis = 28*28

permute = PermuteColumns()
dataset = MNIST(fname="mnist_salakhutdinov.pkl.gz", which_set='salakhutdinov_train', preproc=[permute], n_datapoints=50000)
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

model = LayerStack(
    p_layers=p_layers,
    q_layers=q_layers,
)

trainer = Trainer(
    n_samples=5,
    learning_rate_p=1e-3,
    learning_rate_q=1e-3,
    learning_rate_s=1e-3,
    layer_discount=1.0,
    batch_size=25,
    dataset=dataset, 
    model=model,
    termination=EarlyStopping(min_epochs=250, max_epochs=250),
    #step_monitors=[MonitorLL(data=smallset, n_samples=[1, 5, 25, 100])],
    epoch_monitors=[
        DLogModelParams(), 
        SampleFromP(n_samples=100)
        MonitorLL(name="valiset", data=valiset, n_samples=[100]),
    ],
    final_monitors=[
        MonitorLL(name="final-valiset", data=valiset, n_samples=[1, 5, 10, 25, 100, 500, 1000]),
        MonitorLL(name="final-testset", data=testset, n_samples=[1, 5, 10, 25, 100, 500, 1000]),
    ],
    #monitor_nth_step=100,
)
