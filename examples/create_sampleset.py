''' Reads in PEST parameter files based on glob pattern, creates sampleset, and 
    writes sampleset into a file.
'''
import matk

nms, pars = matk.pest_io.read_par_files( '*.par' )

p = matk.matk()
for n in nms:
    p.add_par( n )

s = p.create_sampleset( pars )

s.savetxt('sampleset.matk')


