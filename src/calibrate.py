from numpy import array,sum,column_stack
from leastsqbound import leastsqbound

def least_squares(myprob):
    mini = []
    maxi = []
    initial_pars = []
    for pargrp in myprob.pargrp:
        for par in pargrp.parameter:
            mini.append(par.min)
            maxi.append(par.max)
            initial_pars.append(par.value)
    mini = array(mini)
    maxi = array(maxi)
    bounds = column_stack([mini,maxi])
    x0 = array(initial_pars)
    res = leastsq_model(initial_pars, myprob)
    #print "Initial SSE: ", sum(res**2)
    x,cov_x,infodic,mesg,ier = leastsqbound( leastsq_model,x0,bounds,args=(myprob),full_output=True)
    res = leastsq_model(x, myprob)
    assert sum( infodic['fvec']**2 ) == sum(res**2), "Calibrated model and current model do not match!"
    #print "Final SSE: ", sum( infodic['fvec']**2 )
    #print infodic['nfev']
    return x,cov_x,infodic,mesg,ier

def leastsq_model( set_pars, args):
    prob = args 
    # set current parameters
    index = 0
    for pargrp in prob.pargrp:
        for par in pargrp.parameter:
            par.value = set_pars[index]
            index += 1
    prob.run_model()
    obs = []
    sims = []
    for obsgrp in prob.obsgrp:
        for observation in obsgrp.observation:
            obs.append( observation.value )
            sims.append( observation.sim_value )
    obs = array(obs)
    sims = array(sims)
    return obs - sims
    