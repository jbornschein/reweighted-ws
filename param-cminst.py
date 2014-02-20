
from learning.dataset import MNIST
from learning.cnade import  CNADE
from learning.training import BatchedSGD
from learning.termination import LogLikelihoodIncrease


dataset = MNIST(which_set='train')

model_params = {
    "unroll_scan": 1,
    "n_vis": 28*28,
    "n_hid": 200,
    "n_cond": 10,
}
model = CNADE(**model_params)


trainer_params = {
    "learning_rate": 0.0005,
    "batch_size": 10,
}
trainer = BatchedSGD(**trainer_params)

termination_param = {}
termination = LogLikelihoodIncrease(**termination_param)

