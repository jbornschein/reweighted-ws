
from mpi4py import MPI
import numpy as np
import unittest

import parallel 


class TestParallel(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
    
    def test_allsort(self):
        D = 2
        my_N = 10
        N = self.comm.size * my_N
        
        my_a = np.random.uniform(size=(my_N, D))
        
        # Check axis=0, 
        all_a = parallel.allsort(my_a, axis=0, kind='quicksort', comm=self.comm)
        self.assertEqual(all_a.shape, (N, D))

        # Chck default axis
        all_a = parallel.allsort(my_a, comm=self.comm)
        self.assertEqual(all_a.shape, (my_N, D*self.comm.size) )

    def test_allargsort(self):
        D = 2
        my_N = 10
        N = self.comm.size * my_N
        
        my_a = np.random.uniform(size=(my_N, D))
        
        # Check axis=0, 
        all_a = parallel.allargsort(my_a, axis=0, kind='quicksort', comm=self.comm)
        self.assertEqual(all_a.shape, (N, D))

        # Chck default axis
        all_a = parallel.allargsort(my_a, comm=self.comm)
        self.assertEqual(all_a.shape, (my_N, D*self.comm.size) )

    def test_allmean(self):
        D = 10
        my_a = np.ones(D)

        mean = parallel.allmean(my_a)
        self.assertAlmostEqual(mean, 1.0)

    def test_allsum(self):
        D = 10
        my_a = np.ones(D)

        sum = parallel.allsum(my_a)
        self.assertAlmostEqual(sum, D*self.comm.size)

    def test_asimetric_gather(self):
        comm = self.comm
        s = comm.size
        #We construct an odd 1-D example to gather
        N = (s-1)*5 + 6

        if comm.rank != (s-1):
            my_N = 5
            my_lb = comm.rank*my_N  
            my_ub = (comm.rank+1)*my_N
        #The last processor takes care of the biggest data part
        else:
            my_N = 6
            my_lb = comm.rank*my_N  
            my_ub = N


        #create the local part of the array (1 and 2 D)
        my_array = np.arange(N)[my_lb: my_ub]
        my_array2 = (np.arange(2*N).reshape(N, 2))[my_lb:my_ub, :]


        #Asimetric gather!
        full_array = np.empty(N, 'int')
        full_array2 = np.empty([N, 2], 'int')

        parallel.asimetric_gather(my_array, N, full_array)
        parallel.asimetric_gather(my_array2, N, full_array2)


        self.assertTrue(np.sum(np.abs(full_array-np.arange(N))) == 0)
        self.assertTrue(np.sum(np.abs(full_array2-(np.arange(2*N).reshape(N, 2)))) == 0)



    def test_sca_assim_load(self):

        comm = self.comm
        s = comm.size

        #We construct a handcrafted example to compare to
        N = 1 + (s-1)*s
        if comm.rank == 0:
            my_lb = 0  
            my_ub = 1
        else:
            my_N = size
            my_lb = (comm.rank-1)*my_N + 1 
            my_ub = my_lb + my_N
        my_load_test = np.arange(my_lb, my_ub)

        #Compare
        my_lb, my_ub = parallel.sca_assim_load(N)
        ##TODO CHeck that this operation is on all the processes as I suppose!
        self.assertTrue(np.sum(np.abs(np.arange(my_lb, my_ub)-my_load_test)) == 0)


    def test_fair_asim_gather(self):
        comm = self.comm
        s = comm.size
        N = s**2 + (s-1)
        out1D = np.empty(N, 'int')
        out2D = np.empty([N, 2], 'int')
        example_2d = np.arange(2*N).reshape(N, 2)

        my_lb, my_ub = parallel.sca_assim_load(N, comm=comm)
        parallel.fair_asim_gather(np.arange(my_lb, my_ub), N, out1D)
        parallel.fair_asim_gather(example_2d[my_lb:my_ub, :], N, out2D)
        
        #Check 1D
        self.assertTrue(np.sum(np.abs(out1D-np.arange(N))) == 0)
        #Check in 2D
        self.assertTrue(np.sum(np.abs(out2D-example_2d)) == 0)
        
