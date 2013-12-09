import sys
import unittest
import matk
from exp_model_int import dbexpl
from sine_decay_model import sine_decay
import numpy

def fv(a):
    ''' Exponential function from marquardt.py
    '''
    a0 = a['a0']
    a1 = a['a1']
    a2 = a['a2']
    X = numpy.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.])
    out = a0 / (1. + a1 * numpy.exp( X * a2))
    return out 

class Tests(unittest.TestCase):

    def setUp(self):
        # Sampling model
        self.p = matk.matk(model=dbexpl)
        self.p.add_par('par1',min=0,max=1)
        self.p.add_par('par2',min=0,max=0.2)
        self.p.add_par('par3',min=0,max=1)
        self.p.add_par('par4',min=0,max=0.2)
        # Calibration model
        # create data to be fitted
        self.x = numpy.linspace(0, 15, 301)
        self.c = matk.matk(model=sine_decay, model_args=(self.x,))
        self.c.add_par('amp', value=5, min=0.)
        self.c.add_par('decay', value=0.025)
        self.c.add_par('shift', value=-0.1, min=-numpy.pi/2., max=numpy.pi/2.)
        self.c.add_par('omega', value=2.0)
        self.c.forward()
        self.c.obs_values = self.c.sim_values
        self.c.par_values = {'amp':10.,'decay':0.1,'shift':0.,'omega':3.0}
        # Model for testing jacobian
        self.j = matk.matk(model=fv)
        self.j.add_par('a0', value=0.7)
        self.j.add_par('a1', value=10.)
        self.j.add_par('a2', value=-0.4)

    def forward(self):
        self.p.forward()
        results = self.p.sim_values
        self.p.obs_values = results
        self.assertEqual(sum(self.p.residuals),0.0, 'Residual from forward run is not zero')

    def sample(self):
        # Create 100 lhs samples and make sure they are within parameter bounds
        self.p.set_lhs_samples('lhs', siz=10)
        s = self.p.sampleset['lhs'].samples
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        lb = self.p.par_mins
        ub = self.p.par_maxs
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Sample outside parameter bounds' )

    def parallel(self):
        # Without working directories
        self.p.set_lhs_samples('lhs', siz=10 )
        self.p.run_samples('lhs', ncpus=2, save=False)
        for smp,out in zip(self.p.sampleset['lhs'].samples,self.p.sampleset['lhs'].responses):
            self.p.par_values = smp
            self.p.forward()
            self.p.obs_values =  out
            self.assertTrue( sum(self.p.residuals) == 0., 'A parallel run does not match a forward run' )

    def parallel_workdir(self):
        # With working directories
        self.p.set_lhs_samples('lhs', siz=10 )
        self.p.run_samples('lhs', ncpus=2, save=True, workdir_base='workdir')
        # Test to make sure reusing directories works
        self.p.run_samples('lhs', ncpus=2, workdir_base='workdir', save=False, reuse_dirs=True)
        for smp,out in zip(self.p.sampleset['lhs'].samples,self.p.sampleset['lhs'].responses):
            self.p.par_values = smp
            self.p.forward()
            self.p.obs_values = out 
            self.assertTrue( sum(self.p.residuals) == 0., 'A parallel run with a working directory does not match a forward run' )

    def parstudy(self):
        lb = self.p.par_mins
        ub = self.p.par_maxs
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


    def calibrate(self): 
        # Look at initial fit
        self.c.forward()
        sims = self.c.sim_values
        # Calibrate parameters to data, results are printed to screen
        self.c.calibrate(report_fit=False)
        # Look at calibrated fit
        self.c.forward()
        sims = self.c.sim_values
        of = numpy.sum(self.c.residuals)
        self.assertTrue( of < 1.e-12, 'Objective function value is ' + str(of) )

    def jacobian(self):
        # Check condition number
        J = self.j.Jac()
        C = numpy.linalg.cond(J)
        self.assertEqual(C.round(16) , 225.6849012361745395, 'Condition number of Jacobian is incorrect')
        
def suite(case):
    suite = unittest.TestSuite()
    suite.addTest( Tests('setUp') )
    if case == 'base' or case == 'all':
        suite.addTest( Tests('forward') )
        suite.addTest( Tests('sample') )
        suite.addTest( Tests('parstudy') )
        suite.addTest( Tests('calibrate') )
        suite.addTest( Tests('jacobian') )
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


