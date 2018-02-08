import sys,os
try:
    import matk
except:
    try:
        sys.path.append(os.path.join('..','src'))
        import matk
    except ImportError as err:
        print 'Unable to load MATK module: '+str(err)
import numpy
from scipy import arange, randn, exp
from multiprocessing import freeze_support

# Model function
def dbexpl(p):
    t=arange(0,100,20.)
    y =  (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
    #nm =  ['o1','o2','o3','o4','o5']
    #return dict(zip(nm,y))
    return y

def run():
    # Setup MATK model with parameters
    p = matk.matk(model=dbexpl)
    p.add_par('par1',min=0,max=1)
    p.add_par('par2',min=0,max=0.2)
    p.add_par('par3',min=0,max=1)
    p.add_par('par4',min=0,max=0.2)

    # Create LHS sample
    s = p.lhs('lhs', siz=500, seed=1000)

    # Run model with parameter samples
    s.run( cpus=2, outfile='results.dat', logfile='log.dat',verbose=False)

    # Save stats for all parameters and responses, use default quantiles
    s.savestats('sampleset.stats')
    # Save stats just for parameters
    s.samples.savestats('parameters.stats')
    # Save stats just for responses
    s.responses.savestats('responses.stats')
    # Specify quantiles
    s.savestats('sampleset_qs.stats',q=[25,50,75])

# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
    freeze_support()
    run()
