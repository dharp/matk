from problem import *
import re
import numpy

def readpest(filename):
    """ Read a PEST control file and populate objects
    
    First argument must be a PEST input file
    Optional second argument is a pymads pest_prob (pymads.pest_prob)
    """
    f = open(filename, 'r')
    
    if not 'pcf' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
    if not '* ' in f.readline():
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
    pest_prob = Problem(npar,nobs,ntplfile,ninsfile,npargrp=mynpargrp,
                        nobsgrp=mynobsgrp)
    
    for i in range(5): f.readline()
    
    if not '* ' in f.readline():
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
            
    if not '* ' in f.readline():
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
            
    while '* ' not in f.readline():
        pass
    
    for i in range(pest_prob.nobsgrp):
        values = f.readline().split()
        obsgrpnm = values[0]
        pest_prob.addobsgrp(obsgrpnm)
         
    if not '* ' in f.readline():
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
    if not '* ' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
 
    pest_prob.sim_command = f.readline().strip()
    
    if not '* ' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
     
    for i in range(pest_prob.ntplfile):
        values = f.readline().split()
        pest_prob.addtpl(values[0], values[1])
        
    for i in range(pest_prob.ninsfile):
        values = f.readline().split()
        pest_prob.addins(values[0], values[1])
 
    return pest_prob


def write_model_files(prob):
    """ Write model from pest template file using current values
    """
    for tplfl in prob.tplfile:
        model_file_str = ''
        for line in tplfl.lines:
            model_file_str += line
        for pargp in prob.pargrp:
            for par in pargp.parameter:
                pattern = '!.*' + par.name + '.*'
                model_file_str = re.sub(r'!.*' + par.name + '.*!', 
                                        str(par.value[-1]), model_file_str)
        f = open( tplfl.modelflname, 'w')
        f.write(model_file_str)
        
def read_model_files(prob):
    """ Collect simulated values from model files using
        pest instruction file
    """
    for insfl in prob.insfile:
        line_index = -1
        f = open( insfl.modelflname , 'r')
        model_file_lines = numpy.array(f.readlines())
        for line in insfl.lines:
            col_index = 0
            values = line.split()
            for val in values:
                if re.match('l', val):
                    line_index += int(re.sub("l","", val))
                if re.match('w', val):
                    col_index += 1
                if re.match('!', val):
                    obsnm = re.sub("!","", val)
                    for obsgrp in prob.obsgrp:
                        for obs in obsgrp.observation:
                            if obs.name == obsnm:
                                values = model_file_lines[line_index].split()
                                obs.sim_value = values[col_index]
                                
def main(argv=None):
    import sys
    if argv is None:
        argv = sys.argv
    pest_prob = readpest(argv[1])
    print pest_prob

if __name__ == "__main__":
    main()
 