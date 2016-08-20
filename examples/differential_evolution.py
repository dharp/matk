from matk import matk
from scipy.optimize import rosen
import numpy as np

def myrosen(pars):
        return rosen(pars.values())

p = matk(model=myrosen)

p.add_par('p1',min=0,max=2)
p.add_par('p2',min=0,max=2)
p.add_par('p3',min=0,max=2)
p.add_par('p4',min=0,max=2)
p.add_obs('o1',value=0)

result = p.differential_evolution()

print "Rosenbrock problem:"
print "Parameters should be all ones: ", result.x
print "Objective function: ", result.fun

def ackley(pars):
    x = pars.values()
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

p2 = matk(model=ackley)

p2.add_par('p1',min=-5,max=5)
p2.add_par('p2',min=-5,max=5)
p2.add_obs('o1',value=0)
result = p2.differential_evolution()

print "Ackley problem:"
print "Parameters should be zero: ", result.x
print "Objective function: ", result.fun
