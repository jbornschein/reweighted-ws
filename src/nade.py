#!/usr/bin/env python 

from __future__ import division

import sys
sys.path.append("../lib")

import logging
from time import time

import numpy as np

import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from unrolled_scan import unrolled_scan

_logger = logging.getLogger(__name__)

#theano.config.compute_test_value = 'warn'
theano_rng = RandomStreams(seed=2341)

def sigmoid_(x):
    return T.nnet.sigmoid(x)*0.9999 + 0.000005

class NADE:
    def __init__(self, n_vis, n_hid, batch_size=100):
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.batch_size = batch_size

        # 
        c = np.zeros( (n_vis), dtype='float32')         # endoder bias
        b = np.zeros( (n_hid), dtype='float32')         # decoder bias
        #W = np.zeros( (n_vis, n_hid), dtype='float32')  # encoder weights
        #V = np.zeros( (n_hid, n_vis), dtype='float32')  # decoder weights

        # random
        #c = np.random.normal( size=(n_vis)).astype(np.float)         # endoder bias
        #b = np.random.normal( size=(n_hid)).astype(np.float)         # decoder bias
        W = (2*np.random.normal( size=(n_vis, n_hid)).astype("float32")-1) / n_vis  # encoder weights
        V = (2*np.random.normal( size=(n_hid, n_vis)).astype("float32")-1) / n_vis  # decoder weights

        self.b = theano.shared(b, name='b')
        self.b.tag.test_value = b
        self.c = theano.shared(c, name='c')
        self.c.tag.test_value = c
        self.W = theano.shared(W, name='W')
        self.W.tag.test_value = W
        self.V = theano.shared(V, name='V')
        self.V.tag.test_value = V

        self.model_params = [self.b, self.c, self.W, self.V]


    def loglikelihood(self, X):
        batch_size = self.batch_size
        n_vis = self.n_vis
        n_hid = self.n_hid

        b = self.c    # decoder bias
        c = self.b    # encoder bias
        W = self.W    # these are shared_vars
        V = self.V    # these are shared_vars
        
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

        #[a, post], updates = theano.scan(
        #            fn=one_iter,
        #            sequences=[vis.T, W, V.T, b],
        #            outputs_info=[a_init, post_init],
        #        )

        [a, post], updates = unrolled_scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b],
                    outputs_info=[a_init, post_init],
                    unroll=8
                )

        return post[-1,:]
        #------------------------------------------------------------------
        """
        def unrolled_iter(vis_i, Wi, Vi, bi, a, post):
            a, post = one_iter(vis_i[0], Wi[0], Vi[0], bi[0], a, post)
            a, post = one_iter(vis_i[1], Wi[1], Vi[1], bi[1], a, post)
            a, post = one_iter(vis_i[2], Wi[2], Vi[2], bi[2], a, post)
            a, post = one_iter(vis_i[3], Wi[3], Vi[3], bi[3], a, post)
            return a, post

    
        sequences = [vis.T.reshape((n_vis//4,4,batch_size)), 
                     W.reshape((n_vis//4,4,n_hid)), 
                     V.T.reshape((n_vis//4,4,n_hid)), 
                     b.reshape((n_vis//4,2))]

        return post[-1,:]
        """

    def f_sample(self):
        batch_size = self.batch_size
        n_vis = self.n_vis
        n_hid = self.n_hid

        b = self.c    # decoder bias
        c = self.b    # encoder bias
        W = self.W    # these are shared_vars
        V = self.V    # these are shared_vars

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
        
        [a, vis], updates = theano.scan(
                    fn=one_iter,
                    sequences=[W, V.T, b], 
                    outputs_info=[a_init, vis_init],
                )

        #theano.printing.debugprint(vis)
        return theano.function([], vis[:,:].T, allow_input_downcast=True)
       
if __name__ == "__main__":
    import timeit

    logging.basicConfig(level=logging.DEBUG)
    _logger = logging.getLogger("nade.py")

    n_vis, n_hid = 8, 3
    nade = NADE(n_vis=n_vis, n_hid=n_hid, batch_size=10)
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
