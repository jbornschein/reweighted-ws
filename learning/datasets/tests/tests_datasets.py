import unittest

import numpy as np

from learning.models.rws import LayerStack
from learning.models.sbn import SBN, SBNTop

# unit under test
from learning.datasets import *
from learning.datasets.mnist import MNIST
from learning.datasets.caltech import CalTechSilhouettes
from learning.datasets.tfd import TorontoFaceDataset

def skip_check(reason):
    raise unittest.SkipTest(reason)

def check_dtype(d):
    assert d.X.dtype == np.float32, "Failed for %s" % d
    assert d.X.ndim == 2
    #if d.Y is not None:
    #    assert d.Y.dtype == np.float32, "Failed for %s" % d
    #    assert d.Y.ndim == 2

def check_same_N(d):
    N = d.n_datapoints
    assert d.X.shape[0] == N, "Failed for %s" % d
    if d.Y is not None:
        assert d.Y.shape[0] == N, "Failed for %s" % d

def check_range(d):
    assert d.X.min() >= 0., "Failed for %s" % d
    assert d.X.max() <= 1., "Failed for %s" % d
    #if d.Y is not None:
    #    assert d.Y.min() >= 0., "Failed for %s" % d
    #    assert d.Y.max() <= 1., "Failed for %s" % d

#-----------------------------------------------------------------------------

test_matrix = {
    (ToyData, 'train')            : (check_dtype, check_same_N, check_range),
    (ToyData, 'valid')            : (check_dtype, check_same_N, check_range),
    (ToyData, 'test' )            : (check_dtype, check_same_N, check_range),
    (BarsData, 'train')           : (check_dtype, check_same_N, check_range),
    (BarsData, 'valid')           : (check_dtype, check_same_N, check_range),
    (BarsData, 'test' )           : (check_dtype, check_same_N, check_range),
    (MNIST, 'train')              : (check_dtype, check_same_N, check_range),
    (MNIST, 'valid')              : (check_dtype, check_same_N, check_range),
    (MNIST, 'test' )              : (check_dtype, check_same_N, check_range),
    (TorontoFaceDataset, 'train') : (check_dtype, check_same_N, check_range),
    (TorontoFaceDataset, 'valid') : (check_dtype, check_same_N, check_range),
    (TorontoFaceDataset, 'test' ) : (check_dtype, check_same_N, check_range),
    (CalTechSilhouettes, 'train') : (check_dtype, check_same_N, check_range),
    (CalTechSilhouettes, 'valid') : (check_dtype, check_same_N, check_range),
    (CalTechSilhouettes, 'test' ) : (check_dtype, check_same_N, check_range),
}

def test_datasets():
    for (ds_class, which_set), tests in test_matrix.iteritems():
        try:
            data = ds_class(which_set=which_set)
        except IOError:
            data = None
           
        for a_check in tests:
            if data is None:
                yield skip_check, ("Could not load %s - IOError" % ds_class)
            else:
                yield a_check, data
 
#-----------------------------------------------------------------------------

def test_FromModel():
    D = 5
    n_vis = D**2
    n_hid = 2*D

    # Ground truth params
    W_bars = np.zeros([n_hid, D, D])
    for d in xrange(D):
        W_bars[  d, d, :] = 4.
        W_bars[D+d, :, d] = 4.
    W_bars = W_bars.reshape( (n_hid, n_vis) )
    P_a = -np.log(D/2-1)*np.ones(n_hid)
    P_b = -2*np.ones(n_vis)
    
    # Instantiate model...
    p_layers = [
        SBN(
            n_X=n_vis, 
            n_Y=n_hid,
        ),
        SBNTop(
            n_X=n_hid
        )
    ]
    q_layers = [
        SBN(
            n_X=n_hid, 
            n_Y=n_vis,
        )
    ]
    p_layers[0].set_model_param('W', W_bars)
    p_layers[0].set_model_param('b', P_b)
    p_layers[1].set_model_param('a', P_a)

    model = LayerStack(
        p_layers=p_layers,
        q_layers=q_layers
    )
    
    # ...and generate data
    n_datapoints = 1000
    data = FromModel(model=model, n_datapoints=n_datapoints)
    
    assert data.X.shape == (n_datapoints, n_vis)
    
    yield check_dtype, data
    yield check_range, data

