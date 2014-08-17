
import numpy as np

from learning.dataset import BarsData, FromH5
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams

from learning.rws import LayerStack
from learning.sbn import SBN, SBNTop
from learning.darn import DARN, DARNTop
from learning.nade import NADE, NADETop

n_vis = 5*5
n_hid = 15
n_qhid = 2*n_hid

dataset = FromH5(fname="deep-bars-5x5-a.h5", n_datapoints=5000)
valiset = FromH5(fname="deep-bars-5x5-a.h5", n_datapoints=1000, offset=5000)
testset = FromH5(fname="deep-bars-5x5-a.h5", n_datapoints=5000, offset=6000)

p_layers=[
    SBN( 
        n_X=n_vis,
        n_Y=15,
    ),
    SBN( 
        n_X=15,
        n_Y=7,
    ),
    SBNTop(
        n_X=7,
    )
]

q_layers=[
    SBN(
        n_X=15,
        n_Y=25,
    ),
    SBN(
        n_X=7,
        n_Y=15,
    ),
]

model = LayerStack(
    p_layers=p_layers,
    q_layers=q_layers,
)

trainer = Trainer(
    n_samples=5,
    learning_rate_p=3e-2,
    learning_rate_q=3e-2,
    learning_rate_s=3e-2,
    layer_discount=1.00,
    batch_size=25,
    dataset=dataset, 
    model=model,
    termination=EarlyStopping(),
    #monitor_nth_step=100,
    #step_monitors=[
    #    MonitorLL(name="valiset", data=valiset, n_samples=[1, 5, 25, 100])
    #],
    epoch_monitors=[
        DLogModelParams(),
        MonitorLL(name="valiset", data=valiset, n_samples=[1, 5, 25, 100]),
        MonitorLL(name="testset", data=testset, n_samples=[1, 5, 25, 100]),
    ],
    final_monitors=[
        MonitorLL(name="final-valiset", data=valiset, n_samples=[1, 5, 25, 100]),
        MonitorLL(name="final-testset", data=testset, n_samples=[1, 5, 25, 100]),
        SampleFromP(data=valiset, n_samples=100),
    ],
)
