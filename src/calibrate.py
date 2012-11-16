from numpy import array,sum,column_stack
from leastsqbound import leastsqbound

def least_squares(myprob):
    mini = array(myprob.get_lower_bounds())
    maxi = array(myprob.get_upper_bounds())
    bounds = column_stack([mini,maxi])
    x0 = array(myprob.get_parameters())
    res = leastsq_model(x0, myprob)
    print "Initial SSE: ", sum(res**2)
    x,cov_x,infodic,mesg,ier = leastsqbound( leastsq_model,x0,bounds,args=(myprob),full_output=True)
    res = leastsq_model(x, myprob)
    assert sum( infodic['fvec']**2 ) == sum(res**2), "Calibrated model and current model do not match!"
    print "Final SSE: ", sum( infodic['fvec']**2 )
    return x,cov_x,infodic,mesg,ier

def leastsq_model( set_pars, args):
    prob = args 
    # set current parameters
    prob.set_parameters(set_pars)
    prob.run_model()
    return array(prob.get_residuals())
    