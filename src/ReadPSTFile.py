import Problem

def readpst(filename,problem):
    """ Read a PEST control file and populate objects
    
    Argument must be a PEST input file
    """
    f = open(filename, 'r')
    #if not 'pcf' in f.readline():
    #    print "%s doesn't appear to be a PEST control file"
    #    return 0
    
    #if not 'control data' in f.readline():
    #    print "%s doesn't appear to be a PEST control file"
    #    return 0
    
    for i in range(3): 
        f.readline()
        
    line = f.readline()
    values = line.split()
    __problem = Problem
    __problem = problem
    problem.npar(values[0])
    problem.nobs(values[1])
    problem.npargp(values[2])
    problem.nobsgp(values[4])
    
     
    if not 'parameter groups' in f.readline():
        print "%s doesn't appear to be a PEST control file"
        return 0
    
    