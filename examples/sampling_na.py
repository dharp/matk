import matk
import numpy
from scipy import arange, randn, exp
try:
    from collections import OrderedDict as dict
except:
    print "Warning: collections module is not installed"
    print "Ordering of observations will not be maintained in output"


# Model function
def dbexpl(p):
    t=arange(0,100,20.)
    if p['par1']<0.5:
        asdf
    y =  (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
    nm =  ['o1','o2','o3','o4','o5']
    return dict(zip(nm,y))


# Setup MATK model with parameters
p = matk.matk(model=dbexpl)
p.add_par('par1',min=0,max=1)
p.add_par('par2',min=0,max=0.2)
p.add_par('par3',min=0,max=1)
p.add_par('par4',min=0,max=0.2)

# Create LHS sample
p.set_lhs_samples('lhs', siz=100, seed=1000)

# Look at sample parameter histograms, correlations, and panels
p.sampleset['lhs'].samples.hist(ncols=2,title='Parameter Histograms',tight=True)
parcor = p.sampleset['lhs'].samples.corr(plot=True, title='Parameter Correlations')
p.sampleset['lhs'].samples.panels(title='Parameter Panels')

# Run model with parameter samples
p.sampleset['lhs'].run( ncpus=2, outfile='results.dat', logfile='log.dat',verbose=False)

# Look at sample response histograms, correlations, and panels
p.sampleset['lhs'].responses.hist(ncols=3,title='Model Response Histograms',tight=True)

# Copy sampleset and subset to only samples with nan responses
p.copy_sampleset('lhs','nans')
p.sampleset['nans'].subset(numpy.isnan, obs='o1')
#p.sampleset['nans'].subset([('o1','numpy.isnan')])

# Evaluate parameter combination resulting in nans
# Note that it is easy to identify that the culprit is par1 with values less than 0.5
p.sampleset['nans'].samples.hist(ncols=2,title='NAN Parameter Histograms',tight=True)
parcor = p.sampleset['nans'].samples.corr(plot=True, title='NAN Parameter Correlations')
p.sampleset['nans'].samples.panels(title='NAN Parameter Panels')

