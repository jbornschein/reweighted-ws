import unittest

# unit under test
from dataset import *

datasets_to_test = [ToyData, MNIST]
array_pairs = [ ('train_x', 'train_y'), 
                ('valid_x', 'valid_y'), 
                ('test_x', 'test_y') ]

def check_dtype(ds):
    for x, y in array_pairs:
        for attr in [x, y]:
            array = getattr(ds, attr, None)
            if array is None:
                continue
            assert array.dtype == np.float32, "Failed for %s %s" % (ds, attr)

def check_same_N(ds):
    for attr_x, attr_y in array_pairs:
        x = getattr(ds, attr_x, None)
        y = getattr(ds, attr_y, None)
        if x is None or y is None:
            continue
    
        assert x.shape[0] == y.shape[0], "Failed for %s %s" % (ds, attr_x)

def check_range(ds):
    for x, y in array_pairs:
        for attr in [x, y]:
            array = getattr(ds, attr, None)
            if array is None:
                continue
            
            assert array.min() >= 0., "Failed for DS %s %s" % (ds, attr)
            print array.max()
            assert array.max() <= 1., "Failed for DS %s %s" % (ds, attr)

def test_dtype():
    for DS in datasets_to_test:
        ds = DS()

        yield check_dtype, ds
        yield check_same_N, ds
        yield check_range, ds
 
