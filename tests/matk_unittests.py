import os,sys
import unittest
sys.path.insert(0,os.path.join('..','src'))
import matk
from exp_model_int import dbexpl
from sine_decay_model import sine_decay
import numpy
from cPickle import dump, load, PicklingError
from scipy.optimize import rosen
import math

def Ishigami(values):
    Y = numpy.zeros([values.shape[0]])
    A = 7
    B = 0.1
    for i, X in enumerate(values):
        Y[i] = math.sin(X[0]) + A * math.pow(math.sin(X[1]), 2) + \
            B * math.pow(X[2], 4) * math.sin(X[0])
	return Y

def myIshigami(pars):
    return Ishigami(numpy.array([[pars['x1'],pars['x2'],pars['x3']]]))

def fv(a):
    ''' Exponential function from marquardt.py
    '''
    a0 = a['a0']
    a1 = a['a1']
    a2 = a['a2']
    X = numpy.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.])
    out = a0 / (1. + a1 * numpy.exp( X * a2))
    return out 

# Define basic function for mcmc
def fmcmc(pars):
    a = pars['a']
    c = pars['c']
    m=numpy.double(numpy.arange(20))
    m=a*(m**2)+c
    return m

#Define basic function for emcee
def femcee(args):
        return numpy.array([args["k"] * 1, args["k"] * 2, args["k"] * 3])

class Tests(unittest.TestCase):

    def setUp(self):
        # Sampling model
        self.p = matk.matk(model=dbexpl)
        self.p.add_par('par1',min=0,max=1)
        self.p.add_par('par2',min=0,max=0.2)
        self.p.add_par('par3',min=0,max=1)
        self.p.add_par('par4',min=0,max=0.2)
        # Calibration sine model
        # create data to be fitted
        self.x = numpy.linspace(0, 15, 301)
        self.c = matk.matk(model=sine_decay, model_args=(self.x,))
        self.c.add_par('amp', value=5, min=0.)
        self.c.add_par('decay', value=0.025)
        self.c.add_par('shift', value=-0.1, min=-numpy.pi/2., max=numpy.pi/2.)
        self.c.add_par('omega', value=2.0)
        self.c.forward()
        self.c.obsvalues = self.c.simvalues
        self.c.parvalues = {'amp':10.,'decay':0.1,'shift':0.,'omega':3.0}
        # Model for testing jacobian
        self.j = matk.matk(model=fv)
        self.j.add_par('a0', value=0.7)
        self.j.add_par('a1', value=10.)
        self.j.add_par('a2', value=-0.4)

    def testforward(self):
        self.p.forward()
        results = self.p.simvalues
        self.p.obsvalues = results
        self.assertEqual(sum(self.p.residuals),0.0, 'Residual from forward run is not zero')

    def testsample(self):
        # Create 100 lhs samples and make sure they are within parameter bounds
        ss = self.p.lhs(siz=10)
        s = ss.samples.values
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        lb = self.p.parmins
        ub = self.p.parmaxs
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Sample outside parameter bounds' )

    def testdiscretesample(self):
        # Create 100 discrete samples and make sure they adhere to assigned probabilities
        p = matk.matk()
        vals = range(5)
        probs = (.1,.2,.3,.2,.2)
        p.add_par('par1',discrete_vals=(vals,probs))
        ss = p.lhs(siz=1000000)
        for i,prob in enumerate(probs): 
            self.assertTrue( numpy.abs(len(numpy.where(ss.recarray['par1']==i)[0])/1000000. - prob) / prob < 0.01, 'Discrete probability is incorrect' )

    def testdiscreteparstudy(self):
        # Ensure that discrete parameter parstudies are correct
        p = matk.matk()
        vals = range(5)
        probs = (.1,.2,.3,.2,.2)
        p.add_par('par1',discrete_vals=(vals,probs))
        ps = p.parstudy(1)
        self.assertEqual( ps.recarray['par1'][0], 2, 'Discrete parstudy of size 1 is incorrect' )
        ps = p.parstudy(2)
        self.assertEqual( ps.recarray['par1'][0], 0., 'Discrete parstudy of size 2 is incorrect' )
        self.assertEqual( ps.recarray['par1'][1], 4., 'Discrete parstudy of size 2 is incorrect' )
        ps = p.parstudy(3)
        self.assertEqual( ps.recarray['par1'][0], 0., 'Discrete parstudy of size 3 is incorrect' )
        self.assertEqual( ps.recarray['par1'][1], 2., 'Discrete parstudy of size 3 is incorrect' )
        self.assertEqual( ps.recarray['par1'][2], 4., 'Discrete parstudy of size 3 is incorrect' )
        ps = p.parstudy(5)
        self.assertEqual( ps.recarray['par1'][0], 0., 'Discrete parstudy of size 3 is incorrect' )
        self.assertEqual( ps.recarray['par1'][1], 1., 'Discrete parstudy of size 3 is incorrect' )
        self.assertEqual( ps.recarray['par1'][2], 2., 'Discrete parstudy of size 3 is incorrect' )
        self.assertEqual( ps.recarray['par1'][3], 3., 'Discrete parstudy of size 3 is incorrect' )
        self.assertEqual( ps.recarray['par1'][4], 4., 'Discrete parstudy of size 3 is incorrect' )

    def testparallel(self):
        # Without working directories
        ss = self.p.lhs(siz=10 )
        ss.run( cpus=2, save=False, verbose=False)
        for smp,out in zip(ss.samples.values,ss.responses.values):
            self.p.parvalues = smp
            self.p.forward()
            self.p.obsvalues =  out
            self.assertTrue( sum(self.p.residuals) == 0., 'A parallel run does not match a forward run' )

    def testparallel_workdir(self):
        # With working directories
        ss = self.p.lhs(siz=10 )
        ss.run( cpus=2, save=True, verbose=False, workdir_base='workdir')
        # Test to make sure reusing directories works
        ss.run( cpus=2, verbose=False, workdir_base='workdir', save=False, reuse_dirs=True)
        for smp,out in zip(ss.samples.values,ss.responses.values):
            self.p.parvalues = smp
            self.p.forward()
            self.p.obsvalues = out 
            self.assertTrue( sum(self.p.residuals) == 0., 'A parallel run with a working directory does not match a forward run' )

    def testcorrelation(self):
        samples = numpy.array([[  2.79514388e-01,   1.83572352e-01,   1.15954591e-01,   4.64518743e-02],
          [  7.03315739e-01,   7.84390758e-02,   3.01698515e-01,   1.88716879e-01],
          [  6.28705093e-01,   1.09417588e-01,   7.99492817e-01,   3.29320144e-02],
          [  9.12638366e-01,   5.79033575e-02,   2.11340456e-01,   9.12899918e-02],
          [  6.78093963e-03,   9.65827955e-02,   5.46769483e-01,   1.60025355e-01],
          [  5.57984090e-01,   1.40754999e-01,   1.67581806e-02,   1.27378742e-01],
          [  4.76628157e-01,   2.13377376e-02,   6.85030919e-01,   1.14660269e-01],
          [  8.74069237e-01,   1.67045111e-01,   8.73799590e-01,   7.88785508e-02],
          [  3.21965099e-01,   1.32870668e-01,   9.84993837e-01,   1.54219267e-01],
          [  1.90139017e-01,   9.68264397e-04,   4.29314483e-01,   9.04097997e-03]])
        s = self.p.create_sampleset(samples=samples)
        s.run( save=False, verbose=False)
        cor = s.corr(printout=False)
        numpy.set_printoptions(precision=16)
        truecor = numpy.array([[-0.2886237899165263, -0.3351709603865224,  0.2026940592413644,
            -0.1583038087507029,  0.6551351732039064],
          [-0.7055158223412237, -0.6756193472215576, -0.6798276930925701,
             -0.7316977209208033,  0.0728971957076234],
          [ 0.0513490094706376,  0.0127386564821643,  0.2351160518407344,
              0.1062707741035715,  0.7433613835033323],
          [-0.6286817433084496, -0.5937202224731661, -0.6548756831743127,
             -0.6743550366736296,  0.0103233615954467]])
        for t,c in zip(cor.flatten(),truecor.flatten()):
            self.assertTrue(numpy.abs(t-c)<1.e-10, 'Value in correlation matrix does not match')

    def testparstudy(self):
        lb = self.p.parmins
        ub = self.p.parmaxs
        # Test keyword args
        ps = self.p.parstudy( nvals=2)
        s = ps.samples.values
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Parstudy outside parameter bounds' )

    def testfullfact(self):
        lb = self.p.parmins
        ub = self.p.parmaxs
        ff = self.p.fullfact( levels=[2,2,2,2] )
        s = ff.samples.values
        mins = s.min(axis=0)
        maxs = s.max(axis=0)
        self.assertTrue( (maxs >= lb).any() and (mins <= ub).any(), 'Full factorial design outside parameter bounds' )

    def testcalibrate_lmfit(self): 
        # Look at initial fit
        self.c.forward()
        sims = self.c.simvalues
        # Calibrate parameters to data, results are printed to screen
        self.c.lmfit(report_fit=False)
        # Look at calibrated fit
        self.c.forward()
        sims = self.c.simvalues
        self.assertTrue( self.c.ssr < 1.e-10, 'Objective function value is ' + str(self.c.ssr) )

    def testcalibrate_lmfit_mp(self): 
        # Look at initial fit
        self.c.forward()
        sims = self.c.simvalues
        # Calibrate parameters to data, results are printed to screen
        self.c.lmfit(cpus=4,report_fit=False)
        # Look at calibrated fit
        self.c.forward()
        sims = self.c.simvalues
        self.assertTrue( self.c.ssr < 1.e-10, 'Objective function value is ' + str(self.c.ssr) )

    def testcalibrate_lmfit_central(self): 
        # Uses central differences to approximate jacobian
        # Look at initial fit
        self.c.forward()
        sims = self.c.simvalues
        # Calibrate parameters to data, results are printed to screen
        self.c.lmfit(cpus=4,report_fit=False,difference_type='central')
        # Look at calibrated fit
        self.c.forward()
        sims = self.c.simvalues
        self.assertTrue( self.c.ssr < 1.e-10, 'Objective function value is ' + str(self.c.ssr) )

    def testjacobian(self):
        # Check condition number
        J = self.j.Jac()
        C = numpy.linalg.cond(J)
        self.assertTrue(numpy.abs(C - 225.684681059)<1.e-8, 'Condition number ('+str(C)+') of Jacobian is incorrect')

    def testcalibrate(self):
        self.j.obsvalues = [5.308,7.24,9.638,12.866,17.069,23.192,31.443,38.558,50.156,62.948,75.995,91.972]
        self.j.calibrate()
        self.assertTrue( self.j.ssr < 2.587278, 'Final SSR of sine model is incorrect' + str(self.j.ssr) )
        self.c.parvalues = {'amp':10.,'decay':0.1,'shift':0.,'omega':3.0}
        self.c.calibrate()
        self.assertTrue( self.c.ssr < 1.e-27, 'Final SSR of marquardt model is incorrect ' + str(self.c.ssr) )

    def testpickle_test(self):
        # Create sampleset
        ss = self.p.lhs(siz=10 )
        try: dump( self.p, open('test.p', 'wb'))
        except PicklingError as errstr: 
            print "Unable to pickle MATK object: "+errstr
            dumpbool = False
        else:
            dumpbool = True
            try: t = load( open('test.p', 'rb'))
            except UnpicklingError as errstr:
                print "Unable to unpickle MATK object: "+errstr
                loadbool=False
            else:
                loadbool=True
            os.remove('test.p')
        self.assertTrue( dumpbool, 'MATK object cannot be pickled' )
        self.assertTrue( loadbool, 'MATK object cannot be unpickled' )

    def testmcmc(self):
        try:
            import pymc
        except:
            print "\nPymc module not installed"
            print "Skipping mcmc unittest"
            return
        self.m = matk.matk(model=fmcmc)
        # Add parameters with 'true' parameters
        self.m.add_par('a', min=0, max=10, value=2)
        self.m.add_par('c', min=0, max=30, value=5)
        # Run model using 'true' parameters
        self.m.forward()
        # Create 'true' observations with zero mean, 0.5 st. dev. gaussian noise added
        self.m.obsvalues = self.m.simvalues + numpy.random.normal(0,1,len(self.m.simvalues))
        # Run MCMC with 100000 samples burning (discarding) the first 10000
        M = self.m.MCMC(nruns=10000,burn=1000, verbose=-1)
        mean_a = M.trace('a').stats()['mean']
        mean_c = M.trace('c').stats()['mean']
        mean_sig = M.trace('error_std').stats()['mean']
        self.assertTrue( abs(mean_a - 2.) < 0.2, 'Mean of parameter a is not close to 2: mean(a) = ' + str(mean_a) )
        self.assertTrue( abs(mean_c - 5.) < 1., 'Mean of parameter c is not close to 5: mean(c) = ' + str(mean_c) )
        self.assertTrue( abs(mean_sig - 1) < 1., 'Mean of model error std. dev. is not close to 0.1: mean(sig) = ' + str(mean_sig) )

    def testemcee(self):
        self.m = matk.matk(model=femcee)
        self.m.add_par("k", value=.5, min=-10, max=10)
        self.m.obsvalues = numpy.array([1., 2., 3.])
        sampler = self.m.emcee(nwalkers=100, nsamples=1000, burnin=100)
        samples = sampler.chain.reshape((-1,len(self.m.pars)))
        mean = numpy.mean(samples)
        std = numpy.std(samples)
        self.assertTrue( abs(mean - 1.) < 0.1, 'Mean of parameter a is not close to 1: mean(samples) = ' + str(mean) )
        self.assertTrue( abs(std - 0.267) < 0.0267, 'Standard deviation is not close to 0.267: std(samples) = ' + str(std) )

    def testemcee2(self):
        self.m = matk.matk(model=fmcmc)
        # Add parameters with 'true' parameters
        self.m.add_par('a', min=0, max=10, value=2)
        self.m.add_par('c', min=0, max=30, value=5)
        # Run model using 'true' parameters
        self.m.forward()
        # Create 'true' observations with zero mean, 0.5 st. dev. gaussian noise added
        self.m.obsvalues = self.m.simvalues + numpy.random.normal(0,0.5,len(self.m.simvalues))
        # Run MCMC with 100000 samples burning (discarding) the first 10000
        #lnprob = matk.logposteriorwithvariance(self.m)
        #print lnprob([2., 5., 10.])
        #print lnprob([2., 8., 10.])
        sampler = self.m.emcee(lnprob=matk.logposteriorwithvariance(self.m), nwalkers=10, nsamples=10000, burnin=1000)
        samples = sampler.chain.reshape((-1,len(self.m.pars)))
        #print samples.shape
        mean_a, mean_c = numpy.mean(samples, 0)
        self.assertTrue( abs(mean_a - 2.) < 0.2, 'Mean of parameter a is not close to 2: mean(a) = ' + str(mean_a) )
        self.assertTrue( abs(mean_c - 5.) < 1., 'Mean of parameter c is not close to 5: mean(c) = ' + str(mean_c) )

    def testemceeunknownvariance(self):
        self.m = matk.matk(model=fmcmc)
        # Add parameters with 'true' parameters
        self.m.add_par('a', min=0, max=10, value=2)
        self.m.add_par('c', min=0, max=30, value=5)
        self.m.add_par('var', min=0, max=1)
        # Run model using 'true' parameters
        self.m.forward()
        # Create 'true' observations with zero mean, 0.5 st. dev. gaussian noise added
        self.m.obsvalues = self.m.simvalues + numpy.random.normal(0,0.5,len(self.m.simvalues))
        # Run MCMC with 100000 samples burning (discarding) the first 10000
        pos0 = [[2+numpy.random.normal(0, 1),5+numpy.random.normal(0, 1),0.5+numpy.random.normal(0, 0.1)] for i in range(10)]
        #lnprob = matk.logposteriorwithvariance(self.m)
        #print lnprob([2., 5., 10.])
        #print lnprob([2., 8., 10.])
        sampler = self.m.emcee(lnprob=matk.logposteriorwithunknownvariance(self.m), nwalkers=10, nsamples=10000, burnin=1000, pos0=pos0)
        samples = sampler.chain.reshape((-1,len(self.m.pars)))
        #print samples.shape
        mean_a, mean_c, mean_sig = numpy.mean(samples, 0)
        mean_sig = numpy.sqrt(mean_sig)
        self.assertTrue( abs(mean_a - 2.) < 0.2, 'Mean of parameter a is not close to 2: mean(a) = ' + str(mean_a) )
        self.assertTrue( abs(mean_c - 5.) < 1., 'Mean of parameter c is not close to 5: mean(c) = ' + str(mean_c) )
        self.assertTrue( abs(mean_sig - 0.5) < 0.25, 'Mean of model error std. dev. is not close to 0.5: mean(sig) = ' + str(mean_sig) )

    def testminimize(self):
        def fun(pars): 
            o = (pars['x1'] - 1)**2 + (pars['x2'] - 2.5)**2
            return -o
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        self.m = matk.matk(model=fun)
        self.m.add_par('x1',min=0,value=2)
        self.m.add_par('x2',min=0,value=0)
        self.m.add_obs('obs1',value=0)
        r = self.m.minimize(constraints=cons,options={'eps':1.4901161193847656e-08})
        self.assertTrue( abs(r['x'][0] - 1.4) < 1.e-8, 'Calibrated parameter 1 should be 1.4 but is ' + str(r['x'][0]) )
        self.assertTrue( abs(r['x'][1] - 1.7) < 1.e-8, 'Calibrated parameter 1 should be 1.7 but is ' + str(r['x'][1]) )

    def testdifferential_evolution(self):
        def myrosen(pars):
                return rosen(pars.values())
        self.m = matk.matk(model=myrosen)
        self.m.add_par('p1',min=0,max=2)
        self.m.add_par('p2',min=0,max=2)
        self.m.add_par('p3',min=0,max=2)
        self.m.add_par('p4',min=0,max=2)
        self.m.add_obs('o1',value=0)
        result = self.m.differential_evolution()
        self.assertTrue( result.fun < 1.e-8, 'Objective function of Rosenbrock problem is larger than tolerance of 1.e-8: ' + str(result.fun) )

        def ackley(pars):
            x = pars.values()
            arg1 = -0.2 * numpy.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
            arg2 = 0.5 * (numpy.cos(2. * numpy.pi * x[0]) + numpy.cos(2. * numpy.pi * x[1]))
            return -20. * numpy.exp(arg1) - numpy.exp(arg2) + 20. + numpy.e
        self.m2 = matk.matk(model=ackley)
        self.m2.add_par('p1',min=-5,max=5)
        self.m2.add_par('p2',min=-5,max=5)
        self.m2.add_obs('o1',value=0)
        result2 = self.m2.differential_evolution()
        self.assertTrue( result2.fun < 1.e-8, 'Objective function for Ackley problem is larger than tolerance of 1.e-8: ' + str(result.fun) )

    def testsobol(self):
        # This test is based on the test problem at: http://salib.readthedocs.io/en/latest/getting-started.html#testing-installation
        m = matk.matk(model=myIshigami)
        m.add_par('x1',min=-3.14159265359, max=3.14159265359)
        m.add_par('x2',min=-3.14159265359, max=3.14159265359)
        m.add_par('x3',min=-3.14159265359, max=3.14159265359)
        m.add_obs('res')
        # Generate samples
        ss = m.saltelli(1000)
        # Run model
        ss.run(verbose=False)
        # Perform analysis
        Si = ss.sobol('res', print_to_console=False)
        # Test results
        self.assertTrue( numpy.abs(Si['S1'][0] - 0.306)/0.306 < 1.e-2, 'First order sensitivity for parameter x1 should be around 0.306 but is ' + str(Si['S1'][0]) )
        self.assertTrue( numpy.abs(Si['S1'][1] - 0.448)/0.448 < 1.e-2, 'First order sensitivity for parameter x2 should be around 0.448 but is ' + str(Si['S1'][1]) )
        self.assertTrue( numpy.abs(Si['S1'][2])<0.01, 'First order sensitivity for parameter x3 should be a very small number but is ' + str(Si['S1'][2]) )

    def testrbd_fast(self):
        # This test is based on running the SALib example at https://github.com/SALib/SALib/blob/master/examples/rbd_fast/rbd_fast.py
        m = matk.matk(model=myIshigami)
        m.add_par('x1',min=-3.14159265359, max=3.14159265359)
        m.add_par('x2',min=-3.14159265359, max=3.14159265359)
        m.add_par('x3',min=-3.14159265359, max=3.14159265359)
        m.add_obs('res')
        # Generate samples
        ss = m.lhs(siz=1000)
        # Run model
        ss.run(verbose=False)
        # Perform analysis
        Si = ss.rbd_fast('res', print_to_console=False)
        # Test results
        self.assertTrue( numpy.abs(Si['S1'][0] - 0.32)/0.32 < 5.e-1, 'First order sensitivity for parameter x1 should be around 0.306 but is ' + str(Si['S1'][0]) )
        self.assertTrue( numpy.abs(Si['S1'][1] - 0.448)/0.448 < 5.e-1, 'First order sensitivity for parameter x2 should be around 0.448 but is ' + str(Si['S1'][1]) )
        self.assertTrue( numpy.abs(Si['S1'][2])<0.1, 'First order sensitivity for parameter x3 should be a very small number but is ' + str(Si['S1'][2]) )
      
def suite(case):
    suite = unittest.TestSuite()
    if case == 'base' or case == 'all':
        suite.addTest( Tests('testforward') )
        suite.addTest( Tests('testsample') )
        suite.addTest( Tests('testparstudy') )
        suite.addTest( Tests('testfullfact') )
        suite.addTest( Tests('testcalibrate_lmfit') )
        suite.addTest( Tests('testjacobian') )
        suite.addTest( Tests('testcalibrate') )
        suite.addTest( Tests('testcorrelation') )
        suite.addTest( Tests('testpickle_test') )
        suite.addTest( Tests('testmcmc') )
        suite.addTest( Tests('testemcee') )
        suite.addTest( Tests('testemcee2') )
        suite.addTest( Tests('testdifferentialevolution') )
        suite.addTest( Tests('testsobol') )
        suite.addTest( Tests('testrbd_fast') )
        suite.addTest( Tests('testdiscretesampling') )
        suite.addTest( Tests('testdiscreteparstudy') )
    if case == 'parallel' or case == 'all':
        suite.addTest( Tests('testparallel') )
        suite.addTest( Tests('testparallel_workdir') )
    if case == 'mcmc':
        #suite.addTest( Tests('mcmc') )
        suite.addTest( Tests('testemcee2') )
    return suite   

if __name__ == '__main__':
    unittest.main()
    #if len(sys.argv) > 1: case = sys.argv[1]
    #else: 
    #    case = 'all'
    #    print "\nAssuming all tests wanted"
    #    print "USAGE: python matk_unittests.py <option>"
    #    print "Option includes: all, parallel, base\n"
    #runner = unittest.TextTestRunner(verbosity=2)
    #test_suite = suite(case)
    #runner.run (test_suite)


