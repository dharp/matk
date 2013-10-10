from subprocess import call
import os
import pesting
from numpy import array
from copy import deepcopy
from shutil import rmtree
from sys import stdout

# Keep in sync with child function in parallel below!
def run_model(command, internal=False):
    """ Make system call to run model, does not write or read model files
        
        Parameters
        ----------
        command : string
            command line string that runs model
    """
    if os.name == 'posix': # If *nix system
        call(command, shell=True, executable='/bin/tcsh')
    else: # If Windows, not sure if this works, maybe get rid of shell=True
        call(command, shell=True)

def forward(prob):
    """ Run forward model using current value
    
        Parameters:
        -----------
        prob : PyMadsProblem object
        
    """
    if not prob.flag['internal']:
        prob.write_model_files()
        prob._run_model()
        prob.read_model_files()
    else:
        prob._run_model()

def parallel(prob, ncpus, par_sets, templatedir=None, workdir_base=None, save_dirs=True ):
    
    try:
        import pp
    except:
        print 'Parallel Python package not available, install to use parallel '
        'execution capability'
        return 1
    
    if templatedir:
        prob.templatedir = templatedir
    if workdir_base:
        prob.workdir_base = workdir_base
    
    def set_child(workdir):
        """ Create working directory for child """
        curdir = os.getcwd()
        tpldir = curdir + '/' + prob.templatedir
        child_dir = curdir + '/' + workdir
        if os.path.exists( child_dir ):
            return 1
        os.makedirs( child_dir )    
        os.chdir( child_dir )
        if not tpldir is None:
            for fil in os.listdir( tpldir ):
                link_file = tpldir + '/' + fil
                link = child_dir + '/' + fil
                os.symlink( link_file, link )
        os.chdir( curdir )
    
    def child( command, child_dir, index ):
        curdir = os.getcwd()
        os.chdir( child_dir )
        if hasattr( command, '__cal__' ):
            out = 'dummy'
        # coped from run_model(command) above, keep in sync!
        if os.name == 'posix': # If *nix system
            subprocess.call(command, shell=True, executable='/bin/tcsh')
        else: # If Windows, not sure if this works, maybe get rid of shell=True
            subprocess.call(command, shell=True)
        os.chdir( curdir )
        return child_dir, index
    
    # Check if a working directory exists
    if not workdir_base is None:
        index = 1
        for par_set in par_sets:
            workdir = prob.workdir_base + '.' + str( index )
            if os.path.exists( workdir ):
                print '\n' + workdir + " already exists!\n"
                return None, None, 1
            index += 1
        
    # Start Parallel Python server
    ppservers = ()
    job_server = pp.Server( ncpus, ppservers=ppservers)
    print "Starting pp with", job_server.get_ncpus(), "workers"

    # Queue model runs
    jobs = []
    index = 1
    for par_set in par_sets:
        prob.set_parameters(par_set)
        if not prob.workdir_base is None:
            child_dir = prob.workdir_base + '.' + str( index )
            st = set_child( child_dir ) 
            if st == 1:
                print "\n" + prob.workdir_base + '.' + str( index ) + " already exists\n"
                return None, None, 1
        prob.write_model_files( workdir=child_dir )
        jobs.append(job_server.submit(child,(prob.model, child_dir, index), (),("os","subprocess",)))
        print "Job ", str(index), " added to queue"
        index += 1
        
    # Wait for jobs and collect results
    responses = []
    for job in jobs:
        child_dir, index = job()
        print "Job ", str(index), " finished"
        prob.read_model_files( workdir=str(child_dir) )
        if not save_dirs:
            rmtree( child_dir )
        responses.append(prob.get_sim_values())
        stdout.flush()
    
    return array(responses), par_sets, 0
        
