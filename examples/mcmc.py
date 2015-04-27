from numpy import array, double, arange, random
try:
    import matk
except:
    try:
        sys.path.append(os.path.join('..','src','matk'))
        import matk
    except ImportError as err:
        print 'Unable to load MATK module: '+str(err)
from multiprocessing import freeze_support

# Define basic function
def f(pars):
    a = pars['a'] 
    c = pars['c'] 
    m=double(arange(20))
    m=a*(m**2)+c
    return m
    
def run():
    # Create matk object
    prob = matk.matk(model=f)

    # Add parameters with 'true' parameters
    prob.add_par('a', min=0, max=10, value=2)
    prob.add_par('c', min=0, max=30, value=5)

    # Run model using 'true' parameters
    prob.forward()

    # Create 'true' observations with zero mean, 0.5 st. dev. gaussian noise added
    prob.obsvalues = prob.simvalues + random.normal(0,0.1,len(prob.simvalues))

    # Run MCMC with 100000 samples burning (discarding) the first 10000
    M = prob.MCMC(nruns=100000,burn=10000)

    # Plot results, PNG files will be created in current directory
    prob.MCMCplot(M)

# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
	freeze_support()
	run()

