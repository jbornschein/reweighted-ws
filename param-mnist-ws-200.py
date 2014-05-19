#
# Wake-sleep experiment with the same parameters as in
#
#   Neural Variational Inference and Learning in Belief Networks (Andriy Mnih, Karol Gregor; 2014)
#   http://arxiv.org/abs/1402.0030
#

import numpy as np

from learning.dataset import BarsData, FromModel, MNIST
from learning.preproc import Binarize
from learning.stbp_layers import  STBPStack, SigmoidBeliefLayer, FactoizedBernoulliTop
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams, SampleFromP

n_vis = 28*28

dataset  = MNIST(fname="data/mnist_salakhutdinov.pkl.gz", which_set='salakhutdinov_train', n_datapoints=59000)
smallset = MNIST(fname="data/mnist_salakhutdinov.pkl.gz", which_set='salakhutdinov_valid', n_datapoints=100)
valiset  = MNIST(fname="data/mnist_salakhutdinov.pkl.gz", which_set='salakhutdinov_valid', n_datapoints=1000)
testset  = MNIST(fname="data/mnist_salakhutdinov.pkl.gz", which_set='test', n_datapoints=10000)

p_layers=[
    SigmoidBeliefLayer( 
        n_X=n_vis,
        n_Y=200,
    ),
    FactoizedBernoulliTop(
        n_X=200,
    )
]

q_layers=[
    SigmoidBeliefLayer(
        n_X=200,
        n_Y=n_vis,
    )
]

model = STBPStack(
    p_layers=p_layers,
    q_layers=q_layers,
)

trainer = Trainer(
    n_samples=1,
    learning_rate_p=1e-4,
    learning_rate_q=0,
    learning_rate_s=2e-5,
    layer_discount=1.0,
    batch_size=20,
    dataset=dataset, 
    model=model,
    termination=EarlyStopping(),
    #step_monitors=[MonitorLL(data=smallset, n_samples=[1, 5, 25, 100])],
    epoch_monitors=[MonitorLL(data=valiset, n_samples=[1, 5, 25, 100]), DLogModelParams(), SampleFromP(n_samples=100)],
    final_monitors=[MonitorLL(data=testset, n_samples=[1, 5, 10, 25, 100, 500])],
    #monitor_nth_step=100,
)
