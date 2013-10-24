import sys
import unittest
import matk
from exp_model_int import dbexpl
import numpy

class Tests(unittest.TestCase):

    def setUp(self):
        self.p = matk.matk(model=dbexpl)
        self.p.add_par('par1',min=0,max=1)
        self.p.add_par('par2',min=0,max=0.2)
        self.p.add_par('par3',min=0,max=1)
        self.p.add_par('par4',min=0,max=0.2)

    def forward(self):
        self.p.forward()
        results = self.p.get_sims()
        self.p.set_obs_values(results)
        self.assertEqual(sum(self.p.get_residuals()),0.0, 'Residual from forward run is not zero')

    def sample(self):
        # Create 100 lhs samples and make sure they are within parameter bounds
        self.p.set_lhs_samples('lhs', siz=10)
        s = self.p.sampleset['lhs'].samples
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        lb = self.p.get_par_mins()
        ub = self.p.get_par_maxs()
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Sample outside parameter bounds' )

    def parallel(self):
        # Without working directories
        self.p.set_lhs_samples('lhs', siz=10 )
        self.p.run_samples('lhs', ncpus=2, save=False)
        for smp,out in zip(self.p.sampleset['lhs'].samples,self.p.sampleset['lhs'].responses):
            self.p.set_par_values( smp )
            self.p.forward()
            self.p.set_obs_values( out )
            self.assertTrue( sum(self.p.get_residuals()) == 0., 'A parallel run does not match a forward run' )

    def parallel_workdir(self):
        # With working directories
        self.p.set_lhs_samples('lhs', siz=10 )
        self.p.run_samples('lhs', ncpus=2, save=True, workdir_base='workdir')
        # Test to make sure reusing directories works
        self.p.run_samples('lhs', ncpus=2, workdir_base='workdir', save=False, reuse_dirs=True)
        for smp,out in zip(self.p.sampleset['lhs'].samples,self.p.sampleset['lhs'].responses):
            self.p.set_par_values( smp )
            self.p.forward()
            self.p.set_obs_values( out )
            self.assertTrue( sum(self.p.get_residuals()) == 0., 'A parallel run with a working directory does not match a forward run' )

    def parstudy(self):
        lb = self.p.get_par_mins()
        ub = self.p.get_par_maxs()
        # Test keyword args
        self.p.set_parstudy_samples( 'ps', par1=2, par2=2, par3=2, par4=2, outfile=None )
        s = self.p.sampleset['ps'].samples
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Parstudy outside parameter bounds' )
        # Test dictionary
        pardict = {'par1':2,'par2':2,'par3':2,'par4':2}
        self.p.set_parstudy_samples( 'ps', pardict )
        s = self.p.sampleset['ps'].samples
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Parstudy outside parameter bounds' )
        # Test list
        s = self.p.set_parstudy_samples( 'ps', (2,2,2,2) )
        s = self.p.sampleset['ps'].samples
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Parstudy outside parameter bounds' )

def suite(case):
    suite = unittest.TestSuite()
    suite.addTest( Tests('setUp') )
    if case == 'base' or case == 'all':
        suite.addTest( Tests('forward') )
        suite.addTest( Tests('sample') )
        suite.addTest( Tests('parstudy') )
    if case == 'parallel' or case == 'all':
        suite.addTest( Tests('parallel') )
        suite.addTest( Tests('parallel_workdir') )
    return suite   

if __name__ == '__main__':
    if len(sys.argv) > 1: case = sys.argv[1]
    else: 
        case = 'all'
        print "\nAssuming all tests wanted"
        print "USAGE: python matk_unittests.py <option>"
        print "Option includes: all, parallel, base\n"
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite(case)
    runner.run (test_suite)


