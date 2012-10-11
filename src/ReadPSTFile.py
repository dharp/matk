from problem import *

def readpst(filename):
    """ Read a PEST control file and populate objects
    
    First argument must be a PEST input file
    Optional second argument is a pymads pest_prob (pymads.pest_prob)
    """
    f = open(filename, 'r')
    
    if not 'pcf' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
    if not 'control data' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
    
    f.readline()
    
    line = f.readline()
    values = line.split()
    npar = int( values[0] )
    nobs = int( values[1] )
    mynpargrp = int( values[2] )
    mynobsgrp = int( values[4] )
    
    line = f.readline()
    values = line.split()
    ntplfile = int( values[0] )
    ninsfile = int( values[1] )
    pest_prob = problem(npar,nobs,ntplfile,ninsfile,npargrp=mynpargrp,
                        nobsgrp=mynobsgrp)
    
    for i in range(5): f.readline()
    
    if not 'parameter groups' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
    
    for i in range(pest_prob.npargrp):
        values = f.readline().split()
        pargrpnm = values[0]
        derinc = values[2]
        derinclb = values[3]
        derincmul = values[5]
        derincmthd = values[6]
        pest_prob.addpargrp(pargrpnm,derinc=derinc,derinclb=derinclb,
                            derincmul=derincmul,derincmthd=derincmthd)
            
    if not 'parameter data' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
 
    for i in range(pest_prob.npar):
        values = f.readline().split()        
        name = values[0]
        initial_value = values[3]
        mn = values[4]
        mx = values[5]
        trans = values[1]
        scale = values[7]
        offset = values[8]
        pargrpnm = values[6]
        parchglim = values[2]
        found = None
        for pgrp in pest_prob.pargrp:
            if pgrp.name == pargrpnm:
                found = True
                pgrp.addparameter(name,initial_value,min=mn,max=mx,
                                  trans=trans,scale=scale,offset=offset,
                                  pargrpnm=pargrpnm,parchglim=parchglim)
        if not found:
            pest_prob.addpargrp(pargrpnm)
            pest_prob.pargrp[-1].addparameter(name,initial_value,min=mn,max=mx,
                          trans=trans,scale=scale,offset=offset,
                          pargrpnm=pargrpnm,parchglim=parchglim)
            
    while 'observation groups' not in f.readline():
        pass
    
    for i in range(pest_prob.nobsgrp):
        values = f.readline().split()
        obsgrpnm = values[0]
        pest_prob.addobsgrp(obsgrpnm)
         
    if not 'observation data' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
  
    for i in range(pest_prob.nobs):
        values = f.readline().split()
        name = values[0]
        value = values[1]
        weight = values[2]
        obsgrpnm = values[3]
        for ogrp in pest_prob.obsgrp:
            if ogrp.name == obsgrpnm:
                found = True
                ogrp.addobservation(name,value,weight=weight,
                                  obsgrpnm=obsgrpnm)
    if not 'model command line' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
 
    pest_prob.sim_command = f.readline().strip()
    
    if not 'model input/output' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
     
    for i in range(pest_prob.ntplfile):
        values = f.readline().split()
        pest_prob.addtpl(values[0], values[1])
        
    for i in range(pest_prob.ninsfile):
        values = f.readline().split()
        pest_prob.addins(values[0], values[1])
 
    return pest_prob

    