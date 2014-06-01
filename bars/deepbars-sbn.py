
import numpy as np

from learning.dataset import BarsData, FromH5
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams

from learning.isws import ISStack
from learning.sbn import SBN, SBNTop
from learning.darn import DARN, DARNTop
from learning.nade import NADE, NADETop

n_vis = 5*5
n_hid = 15
n_qhid = 2*n_hid

#dataset = BarsData(which_set='train', n_datapoints=1000)
#valiset = BarsData(which_set='valid', n_datapoints=100)
dataset = FromH5(fname="deep-bars-5x5-a.h5", n_datapoints=5000)
valiset = FromH5(fname="deep-bars-5x5-a.h5", n_datapoints=1000, offset=5000)


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

model = ISStack(
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
    epoch_monitors=[MonitorLL(data=valiset, n_samples=[1, 5, 25, 100]), DLogModelParams()],
    #step_monitors=[MonitorLL(data=valiset, n_samples=[1, 5, 25, 100])],
    #monitor_nth_step=100
)

