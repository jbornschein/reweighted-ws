#!/usr/bin/env python 

from __future__ import division


import numpy as np

import theano 
import theano.tensor as T

def unrolled_scan(fn, sequences=None, outputs_info=None, non_sequences=None, 
         n_steps=None, truncate_gradient=-1, go_backwards=False, 
         mode=None, name=None, profile=False, unroll=8):
    """ Unrolling version of theano.scan """
    if unroll == 1:
        return theano.scan(fn, sequences=sequences, 
                    outputs_info=outputs_info, 
                    non_sequences=non_sequences,
                    n_steps=n_steps, truncate_gradient=truncate_gradient, 
                    go_backwards= go_backwards, mode=mode, name=name, 
                    profile=profile)

    if sequences is None:
        sequences = []
    if outputs_info is None:
        outputs_info = []
    if non_sequences is None:
        non_sequences = []

    n_seq  = len(sequences)
    n_out  = len(outputs_info)
    n_nseq = len(non_sequences)

    def unrolled_fn(*args):
        if len(args) != (n_seq+n_out+n_nseq):
            raise ValueError('Scan function %s takes %d arguments but expeted to receive %d'
                    % (fn, len(args), (n_seq+n_out+n_nseq)))

        seq_args , args = args[:n_seq], args[n_seq:]
        out_args , args = args[:n_out], args[n_out:]
        nseq_args, args = args[:n_nseq], args[n_nseq:]
        assert len(args) == 0

        
        for i in xrange(unroll):
            seq_args_i = [arg[i] for arg in seq_args]
            all_args = list(seq_args_i)+list(out_args)+list(nseq_args)
            out_args = fn(*all_args)

            if not isinstance(out_args, (tuple, list)):     
                out_args = (out_args,)
            assert len(out_args) == n_out
        if len(out_args) == 1:
            out_args = out_args[0]
        return out_args
    
    def reshape_arg(arg):
        new_shape = [arg.shape[0]//unroll, unroll]+[arg.shape[i] for i in xrange(1, arg.ndim)]
        return arg.reshape(new_shape)
        #return arg.reshape( [arg.shape[0]//unroll, unroll] ) # +arg.shape[1:], ndim=arg.ndim+1 )
        #return arg.reshape( [arg.shape[0]//unroll, unroll]+arg.shape[1:], ndim=arg.ndim+1 )
    sequences = [reshape_arg(arg) for arg in sequences]

    if len(sequences) == 0:
        sequences = None
    if len(outputs_info) == 0:
        outputs_info = None
    if len(non_sequences) == 0:
        non_sequences = None

    return theano.scan(unrolled_fn, sequences=sequences, 
        outputs_info=outputs_info, 
        non_sequences=non_sequences,
        n_steps=n_steps, truncate_gradient=truncate_gradient, 
        go_backwards= go_backwards, mode=mode, name=name, 
        profile=profile)


#-----------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from time import time
    import ipdb
    theano.config.exception_verbosity = 'high'

    def benchmark(fn, tries=4, iterations=100):
        t_best = np.inf
        t_worst = 0.

        for t in xrange(tries):
            t0 = time()
            for i in xrange(iterations):
                fn()
            t = (time()-t0) / iterations
            t_best = min(t_best, t)
            t_worst = max(t_worst, t)
        print "  t_best = %f ms    t_worst = %f ms" %(t_best*1000, t_worst*1000)

    #-------------------------------------------------------------------------

    i = T.arange(100)
    A = theano.shared(np.random.normal(size=(10,10)))
    
    def fn1(seq, acc):
        return T.dot(acc, A)
    
    print "-"*78
    print "Unrolled SCAN:"
    outputs, updates = unrolled_scan(fn1, name='fn1',
        sequences=[i], outputs_info=[T.ones_like(A)],
        unroll=10
    )
    f_fn1 = theano.function([], outputs[-1], name='fn1')

    res = f_fn1()
    print res.shape
    print res
    benchmark(f_fn1)

    print "-"*78
    print "Normal SCAN:"
    outputs, updates = theano.scan(fn1, name='fn1',
        sequences=[i], outputs_info=[T.ones_like(A)]
    )
    f_fn1 = theano.function([], outputs[-1], name='fn1')
    
    res = f_fn1()
    print res.shape
    print res
    benchmark(f_fn1)
    
