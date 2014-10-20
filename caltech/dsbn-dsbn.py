
import numpy as np

from learning.dataset import CalTechSilhouettes
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams, SampleFromP
from learning.monitor.bootstrap import BootstrapLL
from learning.training import Trainer

from learning.rws import LayerStack
from learning.sbn import SBN, SBNTop
from learning.dsbn import DSBN
from learning.darn import DARN, DARNTop
from learning.nade import NADE, NADETop

n_vis = 28*28

dataset  = CalTechSilhouettes(which_set='train')
valiset  = CalTechSilhouettes(which_set='valid')
testset  = CalTechSilhouettes(which_set='test')

p_layers=[
    DSBN( 
        n_X=n_vis,
        n_Y=500,
    ),
    DSBN( 
        n_X=500,
        n_Y=300,
    ),
    DSBN( 
        n_X=300,
        n_Y=100,
    ),
    DSBN( 
        n_X=100,
        n_Y=50,
    ),
    DSBN( 
        n_X=50,
        n_Y=10,
    ),
    SBNTop(
        n_X=10,
    )
]

q_layers=[
    DSBN(
        n_Y=n_vis,
        n_X=500,
    ),
    DSBN(
        n_Y=500,
        n_X=300,
    ),
    DSBN(
        n_Y=300,
        n_X=100,
    ),
    DSBN(
        n_Y=100,
        n_X=50,
    ),
    DSBN(
        n_Y=50,
        n_X=10,
    )
]

model = LayerStack(
    p_layers=p_layers,
    q_layers=q_layers,
)

trainer = Trainer(
    n_samples=10,
    learning_rate_p=1e-4,
    learning_rate_q=1e-4,
    learning_rate_s=1e-4,
    layer_discount=1.0,
    batch_size=100,
    dataset=dataset, 
    model=model,
    termination=EarlyStopping(),
    #step_monitors=[MonitorLL(data=smallset, n_samples=[1, 5, 25, 100])],
    epoch_monitors=[
        DLogModelParams(),
        MonitorLL(name="valiset", data=valiset, n_samples=[1, 5, 25, 100]), 
        SampleFromP(n_samples=100)
    ],
    final_monitors=[
        MonitorLL(name="final-valiset", data=valiset, n_samples=[1, 5, 25, 100, 500, 1000]), 
        MonitorLL(name="final-testset", data=testset, n_samples=[1, 5, 25, 100, 500, 1000]), 
    ],
    monitor_nth_step=100,
)
