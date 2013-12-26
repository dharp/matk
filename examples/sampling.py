import matk
import numpy
from scipy import arange, randn, exp
try:
    from collections import OrderedDict as dict
except:
    print "Warning: collections module is not installed"
    print "Ordering of observations will not be maintained in output"



def dbexpl(p):
    t=arange(0,100,20.)
    y =  (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
    nm =  ['o1','o2','o3','o4','o5']
    return dict(zip(nm,y))


# Sampling model
p = matk.matk(model=dbexpl)
p.add_par('par1',min=0,max=1)
p.add_par('par2',min=0,max=0.2)
p.add_par('par3',min=0,max=1)
p.add_par('par4',min=0,max=0.2)

p.set_lhs_samples('lhs', siz=20, seed=1000)
p.sampleset['lhs'].samples.hist()
p.sampleset['lhs'].run( ncpus=2, outfile='results.dat', logfile='log.dat',verbose=False)
p.sampleset['lhs'].responses.hist(nrows=3)
p.sampleset['lhs'].corr(plot=True) 
p.sampleset['lhs'].corr(plot=True,type='spearman') 

