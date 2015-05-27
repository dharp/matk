# Calibration example modified from lmfit webpage
# (http://cars9.uchicago.edu/software/python/lmfit/parameters.html)
# This example demonstrates how to calibrate with an external code 
# The idea is to replace `python sine.py` in run_extern with any
# terminal command to run your model.
import sys,os
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import freeze_support
from subprocess import Popen,PIPE,call
from matk import matk, pest_io
import cPickle as pickle

def run_extern(params):
    pest_io.tpl_write(params,'../sine.tpl','sine.py')
    ierr = call('python sine.py', shell=True)
    out = pickle.load(open('sine.pkl','rb'))
    return out

# create data to be fitted
x = np.linspace(0, 15, 301)
np.random.seed(1000)
data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
        np.random.normal(size=len(x), scale=0.2) )

# Create MATK object
p = matk(model=run_extern)

# Create parameters
p.add_par('amp', value=10, min=0.)
p.add_par('decay', value=0.1)
p.add_par('shift', value=0.0, min=-np.pi/2., max=np.pi/2.)
p.add_par('omega', value=3.0)

# Create observation names and set observation values
for i in range(len(data)):
    p.add_obs('obs'+str(i+1), value=data[i])

# Look at initial fit
init_vals = p.forward(workdir='initial',reuse_dirs=True)
f, (ax1,ax2) = plt.subplots(2,sharex=True)
ax1.plot(x,data, 'k+')
ax1.plot(x,p.simvalues, 'r')
ax1.set_ylabel("Model Response")
ax1.set_title("Before Calibration")

# Calibrate parameters to data, results are printed to screen
p.lmfit(cpus=2,workdir='calib')

# Look at calibrated fit
ax2.plot(x,data, 'k+')
ax2.plot(x,p.simvalues, 'r')
ax2.set_ylabel("Model Response")
ax2.set_xlabel("x")
ax2.set_title("After Calibration")
plt.show()


