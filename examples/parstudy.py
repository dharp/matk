import sys,os
try:
    import matk
except:
    try:
        sys.path.append(os.path.join('..','src'))
        import matk
    except ImportError as err:
        print 'Unable to load MATK module: '+str(err)
import numpy
from scipy import arange, randn, exp
from multiprocessing import freeze_support

# Model function
def dbexpl(p):
    t=arange(0,100,20.)
    y =  (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
    return y

def run():
	# Setup MATK model with parameters
	p = matk.matk(model=dbexpl)
	p.add_par('par1',min=0,max=1)
	p.add_par('par2',min=0,max=0.2)
	p.add_par('par3',min=0,max=1)
	p.add_par('par4',min=0,max=0.2)
	
	# Create full factorial parameter study with 3 values for each parameter
	s = p.parstudy(nvals=[3,3,3,3])
	
    # Print values to make sure you got what you wanted
	print "\nParameter values:"
	print s.samples.values

	# Look at sample parameter histograms
	s.samples.hist(ncols=2,title='Parameter Histograms by Counts')
	s.samples.hist(ncols=2,title='Parameter Histograms by Frequency',frequency=True)
	
	# Run model with parameter samples
	s.run( cpus=2, outfile='results.dat', logfile='log.dat',verbose=False)
	
	# Look at response histograms, correlations, and panels
	s.responses.hist(ncols=2, bins=30, title='Model Response Histograms')
	rescor = s.responses.corr(plot=True, title='Model Response Correlations')
	s.responses.panels(title='Response Panels')
	
	# Print and plot parameter/response correlations
	print "\nPearson Correlation Coefficients:"
	pcorr = s.corr(plot=True,title='Pearson Correlation Coefficients') 
	print "\nSpearman Correlation Coefficients:"
	scorr = s.corr(plot=True,type='spearman',title='Spearman Rank Correlation Coefficients') 
	s.panels(figsize=(10,8))

# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
	freeze_support()
	run()
