
import numpy as np

from learning.dataset import BarsData, FromModel, MNIST
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams, SampleFromP
from learning.monitor.bootstrap import BootstrapLL

from learning.rws import LayerStack
from learning.sbn import SBN, SBNTop
from learning.darn import DARN, DARNTop
from learning.nade import NADE, NADETop

n_vis = 5*5
n_hid = 15
n_qhid = 2*n_hid

dataset = BarsData(which_set='train', n_datapoints=10000)
valiset = BarsData(which_set='valid', n_datapoints=1000)
testset = BarsData(which_set='test' , n_datapoints=10000)

p_layers=[
    SBN(      
        n_X=n_vis,
        n_Y=n_hid,
    ),
    SBNTop(
        n_X=n_hid,
    )
]

q_layers=[
    SBN(
        n_X=n_hid,
        n_Y=n_vis,
#        n_hid=n_qhid,
#        unroll_scan=1        
    )
]

model = LayerStack(
    p_layers=p_layers,
    q_layers=q_layers,
)

trainer = Trainer(
    n_samples=10,
    learning_rate_p=1e-1,
    learning_rate_q=1e-1,
    learning_rate_s=1e-1,
    batch_size=10,
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
        #BootstrapLL(name="valiset-bootstrap", data=valiset, n_samples=[1, 5, 25, 100])
    ],
    final_monitors=[
        MonitorLL(name="final-valiset", data=valiset, n_samples=[1, 5, 25, 100]),
        MonitorLL(name="final-testset", data=testset, n_samples=[1, 5, 25, 100]),
        SampleFromP(data=valiset, n_samples=100),
    ],
)

