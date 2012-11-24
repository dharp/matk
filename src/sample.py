from lhs import *
from numpy import array

def get_samples(prob, siz=100, noCorrRestr=False, corrmat=None):
    # Take distribution keyword and convert to scipy.stats distribution object
    dists = []
    for dist in prob.get_dists():
        eval( 'dists.append(stats.' + dist + ')' )
    dist_pars = prob.get_dist_pars()
    return lhs(dists, dist_pars, siz, noCorrRestr, corrmat)

def run_samples(prob, siz=100, noCorrRestr=False, corrmat=None, samples=None, file=None):
    if samples == None:
        samples = prob.get_samples(siz)
    out = []
    if not prob.flag['parallel']:
        for sample in samples:
            prob.set_parameters(sample)
            prob.run_model()
            if prob.flag['pest']:
                responses = prob.get_sim_values()
            out.append( responses )
    return array( out ), array( samples )
                
            
    
        