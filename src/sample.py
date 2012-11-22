from lhs import *

def get_samples(prob, siz=100, noCorrRestr=True, corrmat=None):
    # Take distribution keyword and convert to scipy.stats distribution object
    dists = []
    for dist in prob.get_dists():
        eval( 'dists.append(stats.' + dist + ')' )
    dist_pars = prob.get_dist_pars()
    return lhs(dists, dist_pars, siz, noCorrRestr, corrmat)

def sample(prob):
    samples = prob.get_samples()
    if not prob.flag['parallel']:
        for sample in samples:
            prob.set_parameters(sample)
            prob.run_model()
            
    
        