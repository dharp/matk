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
    elif not prob.templatedir:
        print 'Template directory not designated, use optional argument "templatedir"'
        return
    if workdir_base:
        prob.workdir_base = workdir_base
    elif not prob.workdir_base:
        print 'Working directory base not designated, use optional argument "workdir_base"'
        return
    
    def set_child(tpldir, workdir):
        """ Create working directory for child """
        curdir = os.getcwd()
        tpldir = curdir + '/' + prob.templatedir
        child_dir = curdir + '/' + workdir
        if os.path.exists( child_dir ):
            return 1
        os.makedirs( child_dir )    
        os.chdir( child_dir )
        for fil in os.listdir( tpldir ):
            link_file = tpldir + '/' + fil
            link = child_dir + '/' + fil
            os.symlink( link_file, link )
        os.chdir( curdir )
    
    def child( command, child_dir ):
        curdir = os.getcwd()
        os.chdir( child_dir )
        # coped from run_model(command) above, keep in sync!
        if os.name == 'posix': # If *nix system
            subprocess.call(command, shell=True, executable='/bin/tcsh')
        else: # If Windows, not sure if this works, maybe get rid of shell=True
            subprocess.call(command, shell=True)
        os.chdir( curdir )
        return child_dir
    
    # Check if a working directory exists
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
        child_dir = prob.workdir_base + '.' + str( index )
        st = set_child( prob.templatedir, child_dir ) 
        if st == 1:
            print "\nA child directory already exists\n"
            return None, None, 1
        prob.write_model_files( workdir=child_dir )
        jobs.append(job_server.submit(child,(prob.sim_command, child_dir,), (),("os","subprocess",)))
        index += 1
        
    # Wait for jobs and collect results
    prob.flag['sims'] = True
    responses = []
    for job in jobs:
        child_dir = job()
        prob.read_model_files( workdir=str(child_dir) )
        if not save_dirs:
            rmtree( child_dir )
        responses.append(prob.get_sim_values())
        print "Job in ", child_dir, " finished"
        stdout.flush()
    
    return array(responses), par_sets, 0
        
