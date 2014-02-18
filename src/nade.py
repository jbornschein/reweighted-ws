#!/usr/bin/env python 

from __future__ import division

import sys
sys.path.insert(0, "../lib")

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import Model, default_weights
from unrolled_scan import unrolled_scan

_logger = logging.getLogger(__name__)

#theano.config.compute_test_value = 'warn'
theano_rng = RandomStreams(seed=2341)

#------------------------------------------------------------------------------

def sigmoid_(x):
    return T.nnet.sigmoid(x)*0.9999 + 0.000005

#------------------------------------------------------------------------------

class NADE(Model):
    def __init__(self, **hyper_params):
        super(NADE, self).__init__()

        self.register_hyper_param('n_vis', help='no. observed binary variables')
        self.register_hyper_param('n_hid', help='no. latent binary variables')
        self.register_hyper_param('batch_size', default=100)
        self.register_hyper_param('unroll_scan', default=1)

        self.register_model_param('c', help='encoder bias', default=lambda: np.zeros(self.n_vis))
        self.register_model_param('b', help='decoder bias', default=lambda: np.zeros(self.n_hid))
        self.register_model_param('W', help='encoder weights', default=lambda: default_weights(self.n_vis, self.n_hid) )
        self.register_model_param('V', help='decoder weights', default=lambda: default_weights(self.n_hid, self.n_vis) )
        
        self.set_hyper_params(hyper_params)

    def f_loglikelihood(self, X):
        n_vis, n_hid, batch_size = self.get_hyper_params(['n_vis', 'n_hid', 'batch_size'])
        b, c, W, V = self.get_model_params(['b', 'c', 'W', 'V'])
        
        vis = X
        vis.tag.test_value = np.zeros( (batch_size, n_vis), dtype='float32')

        learning_rate = T.fscalar('learning_rate')
        learning_rate.tag.test_value = 0.0

        #------------------------------------------------------------------
        a_init    = T.zeros((batch_size, n_hid), dtype=np.float32) + c
        post_init = T.zeros(batch_size, dtype=np.float32)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = T.nnet.sigmoid(a)
            #pi   = T.nnet.sigmoid(T.dot(hid, Vi) + bi)
            pi   = sigmoid_(T.dot(hid, Vi) + bi)
            post = post + T.cast(T.log(pi*vis_i + (1-pi)*(1-vis_i)), dtype='float32')
            a    = a + T.outer(vis_i, Wi)
            return a, post

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b],
                    outputs_info=[a_init, post_init],
                    unroll=self.unroll_scan
                )
        return post[-1,:]

    def f_sample(self):
        n_vis, n_hid, batch_size = self.get_hyper_params(['n_vis', 'n_hid', 'batch_size'])
        b, c, W, V = get.get_model_params(['b', 'c', 'W', 'V'])

        # 0th iteration 
        a_init   = T.zeros(batch_size)[:,None] + c
        vis_init = T.zeros( (batch_size), dtype=np.float32 )
        
        def one_iter(Wi, Vi, bi, a, vis):
            h = T.nnet.sigmoid(a)
            pi = T.nnet.sigmoid(T.dot(h, Vi) + bi)
            pi = T.shape_padright(pi)
            vi = 1.*(theano_rng.uniform([batch_size,1]) <= pi)
            a  = a + T.outer(vi[:,0], Wi)
            return a, vi.reshape([batch_size])
        
        [a, vis], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[W, V.T, b], 
                    outputs_info=[a_init, vis_init],
                    unroll=self.unroll_scan
                )
        return theano.function([], vis[:,:].T, allow_input_downcast=True)
       
if __name__ == "__main__":
    import timeit

    logging.basicConfig(level=logging.DEBUG)
    _logger = logging.getLogger("nade.py")

    nade = NADE(n_vis=8, n_hid=3)

    f_sample = nade.f_sample()
    f_post = nade.f_post()

    y = np.array(
        [[1., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 1.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 1.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 1.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 1.],
         [1., 1., 1., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 1., 1., 1.]])

    print "="*77
    epochs = 0
    end_learning = False
    LL = [-np.inf]
    t0 = time()
    while not end_learning:
        post, L, gb, gc, gW, gV = f_post(y, 1.0)
        LL.append(L)
        if epochs % 10 == 0:
            print "--- %d ----" % epochs
            print "LL:", L
            #print "post: ", post
            #print "gb:", gb
            #print "b:", nade.b.get_value()
            #print "c:", nade.c.get_value()
        epochs += 1
    
        # Converged?
        end_learning = L <= np.max(LL[-6:-1]) 
        end_learning |= epochs > 10000
    t = time()-t0
    print "Time per epoch: %f" % (t/epochs)
    
    print "="*77
    print "Now drawing samples from this model"

    vis = f_sample()
    print "Samples from model: ", vis

    exit(0)

    f_sample = nade.f_sample()
    vis = f_sample()

    print "=== vis ==="
    print vis
    print vis.shape

    #print timeit.timeit("vis = f_sample()", setup="from __main__ import *", number=100)

    print "="*77
    post, a = f_post(y)

    print "=== post ==="
    print post
    print post.shape

    print "=== a ==="
    print a
    print a.shape

    #print timeit.timeit("vis = f_post(vis)", setup="from __main__ import *", number=100)
