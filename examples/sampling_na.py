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
    #if (p['par1']) < 0.5:
    if (p['par1']+p['par3']) < 0.25:
        asdf
    y =  (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
    return y

def run():
    # Setup MATK model with parameters
    p = matk.matk(model=dbexpl)
    p.add_par('par1',min=0,max=1)
    p.add_par('par2',min=0,max=0.2)
    p.add_par('par3',min=0,max=1)
    p.add_par('par4',min=0,max=0.2)
    
    # Create LHS sample
    s = p.lhs(siz=500, seed=1000)
    
    # Look at sample parameter histograms, correlations, and panels
    s.samples.hist(ncols=2,title='Parameter Histograms')
    parcor = s.samples.corr(plot=True, title='Parameter Correlations')
    s.samples.panels(title='Parameter Panels')
    
    # Run model with parameter samples
    s.run( cpus=2, outfile='results.dat', logfile='log.dat',verbose=False)
    
    # Look at sample response histograms, correlations, and panels
    s.responses.hist(ncols=3,title='Model Response Histograms')
    
    # Copy sampleset and subset to only samples with nan responses
    snan = s.copy()
    snan.subset(numpy.isnan, obs='obs1')
    
    # Evaluate parameter combination resulting in nans
    # Note that it is easy to identify that the culprit is par1 with values less than 0.5
    snan.samples.hist(ncols=2,title='NAN Parameter Histograms')
    parcor = snan.samples.corr(plot=True, title='NAN Parameter Correlations')
    snan.samples.panels(title='NAN Parameter Panels')
    
# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
    freeze_support()
    run()
