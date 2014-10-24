
import numpy as np

from learning.dataset import CalTechSilhouettes
from learning.preproc import PermuteColumns
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams, SampleFromP
from learning.training import Trainer

from learning.models.rws  import LayerStack
from learning.models.sbn  import SBN, SBNTop
from learning.models.darn import DARN, DARNTop
from learning.models.nade import NADE, NADETop

n_vis = 28*28

preproc = PermuteColumns()

dataset  = CalTechSilhouettes(which_set='train', preproc=[preproc])
valiset  = CalTechSilhouettes(which_set='valid', preproc=[preproc])
testset  = CalTechSilhouettes(which_set='test', preproc=[preproc])

p_layers=[
    NADE( 
        n_X=n_vis,
        n_Y=150,
    ),
    NADETop( 
        n_X=150,
    ),
]

q_layers=[
    NADE(
        n_Y=n_vis,
        n_X=150,
    ),
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
