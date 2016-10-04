# Calibration example modified from lmfit webpage
# (http://cars9.uchicago.edu/software/python/lmfit/parameters.html)
# This example demonstrates how to utilize cluster resources
# for calibration with an external simulator.
# The idea is to replace `python sine.py` in run_extern with any
# terminal command to run your model.
# Also note that the hosts dictionary can contain any remote hosts
# accessible by passwordless ssh, for instance other workstations
# on your network.
import sys,os
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import freeze_support
from subprocess import Popen,PIPE,call
from matk import matk, pest_io
import cPickle as pickle

def run_extern(params, hostname=None, processor=None):
    pest_io.tpl_write(params,'../sine.tpl','sine.py')
    ierr = call('ssh '+hostname+" 'cd "+os.getcwd()+" && python sine.py'", shell=True)
    out = pickle.load(open('sine.pkl','rb'))
    return out

# Automatically determine the hostnames available on system using slurm resource manager
# This will have to be modified for other resource managers
hostnames = Popen(["scontrol","show","hostnames"],stdout=PIPE).communicate()[0]
hostnames = hostnames.split('\n')[0:-1]
host = os.environ['HOST'].split('.')[0]
#hostnames.remove(host) # Remove host to use as designated master if desired

# Create dictionary of lists of processor ids to use keyed by hostname
hosts = {}
for h in hostnames:
    hosts[h] = range(0,16,6) # create lists of processor numbers for each host
print 'host dictionary: ', hosts

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
init_vals = p.forward(workdir='initial',hostname=hosts.keys()[0],processor=0,reuse_dirs=True)
plt.plot(x,data, 'k+')
plt.plot(x,p.sim_values, 'r')
plt.title("Before Calibration")
plt.show(block=True)

# Calibrate parameters to data, results are printed to screen
p.lmfit(cpus=hosts,workdir='calib')

# Look at calibrated fit
plt.plot(x,data, 'k+')
plt.plot(x,p.sim_values, 'r')
plt.title("After Calibration")
plt.show()


