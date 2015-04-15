import sys,os
try:
    import matk
except:
    try:
        sys.path.append('..'+os.sep+'..'+os.sep+'src')
        import matk
    except ImportError as err:
        print 'Unable to load MATK module: '+str(err)
import numpy
import pest_io
from subprocess import call
import fpost
from multiprocessing import freeze_support

# Model function
def fehm(p):
    # Create simulator input file
    pest_io.tpl_write(p, '../intact.tpl', 'intact.dat')
    # Call simulator
    ierr = call('xfehm ../intact.files', shell=True)
    # Collect result of interest and return
    o = fpost.fnodeflux('intact.internode_fluxes.out')
    return [o[o.nodepairs[-1]]['liquid'][-1]]

def run():
    # Setup MATK model with parameters
    p = matk.matk(model=fehm)
    p.add_par('por0',min=0.1,max=0.3)

    # Create LHS sample
    s = p.parstudy(nvals=[3])

    # Run model with parameter samples
    s.run( ncpus=2, workdir_base='workdir', outfile='results.dat', logfile='log.dat',verbose=False,reuse_dirs=True)

    # Look at response histograms, correlations, and panels
    print 'Parameter Response'
    for pa,re in zip(s.samples.values, s.responses.values): print pa[0], re[0]
    s.responses.hist(ncols=2,title='Model Response Histograms')

# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
	freeze_support()
	run()

