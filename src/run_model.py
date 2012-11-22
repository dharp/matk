from subprocess import call
from os import name
import pesting
from numpy import array


def forward(prob):
    """ Run forward model using current value
    
        Parameters:
        -----------
        prob : PyMadsProblem object
        
    """
    if prob.flag['pest']:
        pesting.write_model_files(prob)
    if name == 'posix': # If *nix system
        call(prob.sim_command, shell=True, executable='/bin/tcsh')
    else: # If Windows, not sure if this works, maybe get rid of shell=True
        call(prob.sim_command, shell=True)
    if prob.flag['pest']:
        pesting.read_model_files(prob)
    return 0

def parallel(prob):
    try:
        import pp
    except:
        print 'Parallel Python package not available, install to use parallel \
        execution capability'
        return 1
    
    ppservers = ()
    # Determine number of cpus for jobserver
    job_server = pp.Server(prob.ncpus, ppservers=ppservers)

    print "Starting pp with", job_server.get_ncpus(), "workers"