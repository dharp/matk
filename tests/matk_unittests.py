import unittest
import matk
from exp_model_int import dbexpl
from numpy import array
from glob import glob
from shutil import rmtree


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.p = matk.matk(model=dbexpl)
        self.p.add_par('par1',min=0,max=1)
        self.p.add_par('par2',min=0,max=1)
        self.p.add_par('par3',min=0,max=1)
        self.p.add_par('par4',min=0,max=1)

    def test_forward(self):
        self.p.forward()
        results = self.p.get_sims()
        self.p.set_obs_values(results)
        self.assertEqual(sum(self.p.get_residuals()),0.0, 'Residual from forward run is not zero')

    def test_sample(self):
        s = self.p.get_samples(siz=1, noCorrRestr=True)
        lb = self.p.get_par_mins()
        ub = self.p.get_par_maxs()
        self.assertTrue( (s >= lb).any() and (s <= ub).any(), 'Sample outside parameter bounds' )

    def test_parallel(self):
        s = self.p.get_samples(siz=10)
        # Without working directories
        o,s = self.p.run_samples(siz=10, parallel=True, save=False)
        for smp,out in zip(s,o):
            self.p.set_par_values( smp )
            self.p.forward()
            self.p.set_obs_values( out )
            self.assertTrue( sum(self.p.get_residuals()) == 0., 'A parallel run does not match a forward run' )
        # With working directories
        o,s = self.p.run_samples(siz=10, ncpus=2, parallel=True, workdir_base='workdir', save=False)
        for smp,out in zip(s,o):
            self.p.set_par_values( smp )
            self.p.forward()
            self.p.set_obs_values( out )
            self.assertTrue( sum(self.p.get_residuals()) == 0., 'A parallel run does not match a forward run' )

            
        

if __name__ == '__main__':
    unittest.main()

