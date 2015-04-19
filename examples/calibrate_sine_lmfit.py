# Calibration example modified from lmfit webpage
# (http://cars9.uchicago.edu/software/python/lmfit/parameters.html)
import sys,os
try:
    import matk
except:
    try:
        sys.path.append(os.path.join('..','src','matk'))
        import matk
    except ImportError as err:
        print 'Unable to load MATK module: '+str(err)
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import freeze_support

# define objective function: returns the array to be minimized
def sine_decay(params, x, data):
    """ model decaying sine wave, subtract data"""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']

    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)

    obsnames = ['obs'+str(i) for i in range(1,len(data)+1)]
    return dict(zip(obsnames,model))


def run():
    # create data to be fitted
    x = np.linspace(0, 15, 301)
    np.random.seed(1000)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2) )

    # Create MATK object
    p = matk.matk(model=sine_decay, model_args=(x,data,))

    # Create parameters
    p.add_par('amp', value=10, min=0.)
    p.add_par('decay', value=0.1)
    p.add_par('shift', value=0.0, min=-np.pi/2., max=np.pi/2.)
    p.add_par('omega', value=3.0)

    # Create observation names and set observation values
    for i in range(len(data)):
        p.add_obs('obs'+str(i+1), value=data[i])

    # Look at initial fit
    p.forward()
    plt.plot(x,data, 'k+')
    plt.plot(x,p.sim_values, 'r')
    plt.title("Before Calibration")
    plt.show(block=True)

    # Calibrate parameters to data, results are printed to screen
    p.lmfit(cpus=2)

    # Look at calibrated fit
    plt.plot(x,data, 'k+')
    plt.plot(x,p.sim_values, 'r')
    plt.title("After Calibration")
    plt.show()

# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
	freeze_support()
	run()

