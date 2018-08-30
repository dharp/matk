import numpy as np

# define objective function: returns the array to be minimized
def sine_decay(params, x):
    """ model decaying sine wave, subtract data"""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']

    np.seterr(all='ignore')
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)

    obsnames = ['obs'+str(i) for i in range(1,len(model)+1)]
    return dict(list(zip(obsnames,model)))


