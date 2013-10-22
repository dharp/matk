import sys
import unittest
import matk
from exp_model_int import dbexpl

class Tests(unittest.TestCase):

    def setUp(self):
        self.p = matk.matk(model=dbexpl)
        self.p.add_par('par1',min=0,max=1)
        self.p.add_par('par2',min=0,max=1)
        self.p.add_par('par3',min=0,max=1)
        self.p.add_par('par4',min=0,max=1)

    def forward(self):
        self.p.forward()
        results = self.p.get_sims()
        self.p.set_obs_values(results)
        self.assertEqual(sum(self.p.get_residuals()),0.0, 'Residual from forward run is not zero')

    def sample(self):
        s = self.p.get_samples(siz=1, noCorrRestr=True)
        lb = self.p.get_par_mins()
        ub = self.p.get_par_maxs()
        self.assertTrue( (s >= lb).any() and (s <= ub).any(), 'Sample outside parameter bounds' )

    def parallel(self):
        # Without working directories
        o,s = self.p.run_samples(siz=10, ncpus=2, parallel=True, save=False)
        for smp,out in zip(s,o):
            self.p.set_par_values( smp )
            self.p.forward()
            self.p.set_obs_values( out )
            self.assertTrue( sum(self.p.get_residuals()) == 0., 'A parallel run does not match a forward run' )

    def parallel_workdir(self):
        # With working directories
        o,s = self.p.run_samples(siz=10, ncpus=2, parallel=True, workdir_base='workdir', save=False)
        for smp,out in zip(s,o):
            self.p.set_par_values( smp )
            self.p.forward()
            self.p.set_obs_values( out )
            self.assertTrue( sum(self.p.get_residuals()) == 0., 'A parallel run with a working directory does not match a forward run' )

    def parstudy(self):
        lb = self.p.get_par_mins()
        ub = self.p.get_par_maxs()
        # Test keyword args
        s = self.p.get_parstudy( par1=2, par2=2, par3=2, par4=2, outfile=None )
        self.assertTrue( (s >= lb).any() and (s <= ub).any(), 'Parstudy outside parameter bounds' )
        # Test dictionary
        pardict = {'par1':2,'par2':2,'par3':2,'par4':2}
        s = self.p.get_parstudy( pardict )
        self.assertTrue( (s >= lb).any() and (s <= ub).any(), 'Parstudy outside parameter bounds' )
        # Test list
        s = self.p.get_parstudy( (2,2,2,2) )
        self.assertTrue( (s >= lb).any() and (s <= ub).any(), 'Parstudy outside parameter bounds' )

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


