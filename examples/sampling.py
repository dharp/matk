import matk
import numpy
from scipy import arange, randn, exp

def dbexpl(p):
    t=arange(0,100,20.)
    y =  (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
    return {'obs1':y[0],'obs2':y[1],'obs3':y[2],'obs4':y[3],'obs5':y[4]}


# Sampling model
p = matk.matk(model=dbexpl)
p.add_par('par1',min=0,max=1)
p.add_par('par2',min=0,max=0.2)
p.add_par('par3',min=0,max=1)
p.add_par('par4',min=0,max=0.2)

p.set_lhs_samples('lhs', siz=10, seed=1000)
p.sampleset['lhs'].run( ncpus=1, outfile='results.dat')
 
