# Calibration example modified from lmfit webpage
# (http://cars9.uchicago.edu/software/python/lmfit/parameters.html)
import sys,os
try:
    import matk
except:
    try:
        sys.path.append(os.path.join('..','src'))
        import matk
    except ImportError as err:
        print 'Unable to load MATK module: '+str(err)
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import freeze_support
from collections import OrderedDict

def calibrate(params,x,data):
    pc = matk.matk(model=sine_decay,model_args=(x,data,))

    # Create parameters
    pc.add_par('amp', value=10, min=0.)
    pc.add_par('decay', value=0.1)
    pc.add_par('shift', value=0.0, min=-np.pi/2., max=np.pi/2.)
    pc.add_par('omega', value=3.0)

    # Create observation names and set observation values
    for i in range(len(data)):
        pc.add_obs('obs'+str(i+1), value=data[i])

    # Set initial values
    pc.parvalues = params.values()

    # Calibrate
    pc.lmfit(report_fit=False)
    return pc.parvalues.tolist() + [pc.ssr]

# define objective function: returns the array to be minimized
def sine_decay(params, x, data):
    """ model decaying sine wave, subtract data"""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']

    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)

    obsnames = ['obs'+str(i) for i in range(1,len(data)+1)]
    return OrderedDict(zip(obsnames,model))


def run():
    # create data to be fitted
    x = np.linspace(0, 15, 301)
    np.random.seed(1000)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2) )

    # Create MATK object
    p = matk.matk(model=calibrate, model_args=(x,data,))

    # Create parameters
    p.add_par('amp', value=10, min=9., max=11.)
    p.add_par('decay', value=0.1, min=0.09, max=0.11)
    p.add_par('shift', value=0.0, min=-np.pi/6., max=np.pi/6.)
    p.add_par('omega', value=3.0, min=2.5, max=3.5)

    ## Create observation names and set observation values
    #for i in range(len(data)):
    #    p.add_obs('obs'+str(i+1), value=data[i])

    s = p.lhs(siz=30,seed=40)
    s.run(cpus=5)

    best_id = np.argmin(s.responses.values[:,-1])
    print "Lowest objective function value found:"
    print s.responses.values[best_id][-1]
    print "Best parameters found:"
    print s.responses.values[best_id][:-1]
    
    # Look at initial fit
    simvalues = sine_decay(dict(zip(p.parnames,s.samples.values[best_id])),x,data).values()
    f, (ax1,ax2) = plt.subplots(2,sharex=True)
    ax1.plot(x,data, 'k+')
    ax1.plot(x,simvalues, 'r')
    ax1.set_ylabel("Model Response")
    ax1.set_title("Before Calibration")

    # Look at calibrated fit
    simvalues = sine_decay(dict(zip(p.parnames,s.responses.values[best_id][:-1])),x,data).values()
    ax2.plot(x,data, 'k+')
    ax2.plot(x,simvalues, 'r')
    ax2.set_ylabel("Model Response")
    ax2.set_xlabel("x")
    ax2.set_title("After Calibration")
    plt.show(block=True)

# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
    freeze_support()
    run()

