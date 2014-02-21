import unittest

# unit under test
from dataset import *

def skip_check(reason):
    raise unittest.SkipTest(reason)

def check_dtype(d):
    assert d.X.dtype == np.float32, "Failed for %s" % d
    assert d.Y.dtype == np.float32, "Failed for %s" % d
    assert d.X.ndim == 2
    assert d.Y.ndim == 2

def check_same_N(d):
    N = d.n_datapoints
    assert d.X.shape[0] == N, "Failed for %s" % d
    assert d.Y.shape[0] == N, "Failed for %s" % d

def check_range(d):
    assert d.X.min() >= 0., "Failed for %s" % d
    assert d.X.max() <= 1., "Failed for %s" % d
    assert d.Y.min() >= 0., "Failed for %s" % d
    assert d.Y.max() <= 1., "Failed for %s" % d

#-----------------------------------------------------------------------------

test_matrix = {
    (ToyData, 'train') : (check_dtype, check_same_N, check_range),
    (ToyData, 'valid') : (check_dtype, check_same_N, check_range),
    (ToyData, 'test' ) : (check_dtype, check_same_N, check_range),
    (BarsData, 'train'): (check_dtype, check_same_N, check_range),
    (BarsData, 'valid'): (check_dtype, check_same_N, check_range),
    (BarsData, 'test' ): (check_dtype, check_same_N, check_range),
    (MNIST, 'train')   : (check_dtype, check_same_N, check_range),
    (MNIST, 'valid')   : (check_dtype, check_same_N, check_range),
    (MNIST, 'test' )   : (check_dtype, check_same_N, check_range),
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
 
