from lhs import *
from numpy import array

def get_samples(prob, siz=100, noCorrRestr=False, corrmat=None):
    # Take distribution keyword and convert to scipy.stats distribution object
    dists = []
    for dist in prob.get_dists():
        eval( 'dists.append(stats.' + dist + ')' )
    dist_pars = prob.get_dist_pars()
    return lhs(dists, dist_pars, siz, noCorrRestr, corrmat)

def run_samples(prob, siz=100, noCorrRestr=False, corrmat=None,
                 samples=None, outfile=None, parallel=False, ncpus=None,
                  templatedir=None, workdir_base=None):
    if samples == None:
        samples = prob.get_samples(siz)
    if parallel:
        prob.flag['parallel'] = True
        if not ncpus:
            print 'Number of cpus is not set for parallel model runs, use option ncpus'
            return 1
        if templatedir:
            prob.templatedir = templatedir
        elif not prob.templatedir:
            print 'Template directory not designated, use option templatedir'
            return 1
        if workdir_base:
            prob.workdir_base = workdir_base
        elif not prob.workdir_base:
            print 'Working directory base name not designated, use option workdir_base'
            return 1
            
    if not prob.flag['parallel']:
        out = []
        for sample in samples:
            prob.set_parameters(sample)
            prob.run_model()
            if prob.flag['pest']:
                responses = prob.get_sim_values()
            out.append( responses )
    else:
        samples, out = prob.parallel(ncpus, samples, templatedir=templatedir, workdir_base=workdir_base)
        
    return array( out ), array( samples )
                
            
    
        