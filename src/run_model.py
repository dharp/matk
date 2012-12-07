from subprocess import call
import os
import pesting
from numpy import array
from copy import deepcopy


def forward(prob):
    """ Run forward model using current value
    
        Parameters:
        -----------
        prob : PyMadsProblem object
        
    """
    if prob.flag['pest']:
        pesting.write_model_files(prob)
    if os.name == 'posix': # If *nix system
        call(prob.sim_command, shell=True, executable='/bin/tcsh')
    else: # If Windows, not sure if this works, maybe get rid of shell=True
        call(prob.sim_command, shell=True)
    if prob.flag['pest']:
        pesting.read_model_files(prob)
    return 0

def parallel(prob, ncpus, par_sets, templatedir=None, workdir_base=None ):
    
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
    
    def child(pars, prob):
        curdir = os.getcwd()
        tpldir = curdir + '/' + prob.templatedir
        child_dir = curdir + '/' + prob.workdir_base + '.' + str( int(prob.workdir_index) )
        if os.path.exists( child_dir ):
            return 1
        os.makedirs( child_dir )    
        os.chdir( child_dir )
        for fil in os.listdir( tpldir ):
            link_file = tpldir + '/' + fil
            link = child_dir + '/' + fil
            os.symlink( link_file, link )
        prob.set_parameters( pars )
        prob.run_model()
        os.chdir( curdir )
        return prob
    
    ppservers = ()
    
    job_server = pp.Server( ncpus, ppservers=ppservers)

    print "Starting pp with", job_server.get_ncpus(), "workers"
    
    
    # Check if a working directory exists
    index = 1
    for par_set in par_sets:
        workdir = prob.workdir_base + '.' + str( index )
        if os.path.exists( workdir ):
            print '\n' + workdir + " already exists!\n"
            return 1, 1
        index += 1
        
    jobs = []
    child_probs = []
    index = 1
    for par_set in par_sets:
        # Copy prob to child_prob[] using deep copy so that it is its own instance
        child_probs.append(deepcopy(prob))
        child_probs[-1].workdir_index = index
        index += 1
        jobs.append(job_server.submit(child,(par_set, child_probs[-1]), (),("pymads",)))
        
    out = []
    for job in jobs:
        prob = job()
        if prob == 1:
            print "\nA child directory already exists\n"
            return 1
        out.append(prob.get_sim_values())
    
    return par_sets, array(out)
        