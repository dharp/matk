''' Reads in PEST parameter files based on glob pattern, creates sampleset, and 
    writes sampleset into a file.
    Requires PEST parameter files to run
'''
try:
    import matk
except:
    try:
        sys.path.append(os.path.join('..','src'))
        import matk
    except ImportError as err:
        print 'Unable to load MATK module: '+str(err)
from multiprocessing import freeze_support

def run():

    nms, pars = matk.pest_io.read_par_files( '*.par' )

    p = matk.matk()
    for n in nms:
        p.add_par( n )
    
    s = p.create_sampleset( pars )

    s.savetxt('sampleset.matk')


# Freeze support is necessary for multiprocessing on windows
if __name__== "__main__":
    freeze_support()
    run()
