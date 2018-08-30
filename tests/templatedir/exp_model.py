"""
Example of curve fitting for
a1*exp(-k1*t) + a2*exp(-k2*t)
"""
from scipy import arange, randn, exp
#from matplotlib import *
#from pylab import *
#from scipy.optimize import leastsq
#from pymads import *
#from pesting import *

def dbexpl(p):
    t=arange(0,100,10.)
    return(p[0]*exp(-p[1]*t) + p[2]*exp(-p[3]*t))

def create_data():
    a1,a2 = 1.0, 1.0
    k1,k2 = 0.05, 0.2
    t=arange(0,100,10.)
    data = dbexpl(t,[a1,k1,a2,k2]) + 0.02*randn(len(t))
    return data

#def residuals(p,data,t):
#    err = data - dbexpl(t,p)
#    return err
#
def main():
    # Read in parameters
    p = []
    f = open('exp_model.in', 'r')
    for line in f:
        p.append(float(line.strip()))
    f.close()
    #assert len(p) == 4, "Incorrect number of lines in exp_model.in!"
    data = dbexpl(p)
    f = open('exp_model.out', 'w')
    for val in data:
        f.write('{0:.13f}'.format(val) + '\n')
        print(('{0:.13f}'.format(val)))
    f.close()

if __name__ == '__main__':
    main()
    
