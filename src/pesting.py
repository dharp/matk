__all__ = ['read_pest', 'read_model_files', 'write_model_files', 'ModelTemplate', 'ModelInstruction']

import pymads
import re
from numpy import array

def read_pest(filename):
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
    pest_prob = pymads.PyMadsProblem(npar,nobs,ntplfile,ninsfile,npargrp=mynpargrp,
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
 
    pest_prob.flag['pest'] = True
    
    return pest_prob

def read_model_files(prob):
    """ Collect simulated values from model files using
        pest instruction file
    """
    for insfl in prob.insfile:
        line_index = -1
        f = open( insfl.modelflname , 'r')
        model_file_lines = array(f.readlines())
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

def write_model_files(prob):
    """ Write model from pest template file using current values
    """
    for tplfl in prob.tplfile:
        model_file_str = ''
        for line in tplfl.lines:
            model_file_str += line
        for pargp in prob.pargrp:
            for par in pargp.parameter:
                model_file_str = re.sub(tplfl.marker + r'.*' + par.name + r'.*' + tplfl.marker, 
                                        str(par.value), model_file_str)
        f = open( tplfl.modelflname, 'w')
        f.write(model_file_str)
        
class ModelInstruction(object):
    """pymads PEST instruction file class
    """
    def __init__(self,insflname,modelflname):
        self.insflname = insflname
        self.modelflname = modelflname
        f = open( self.insflname, 'r')
        self.lines = f.readlines()
        lines = array(self.lines)
        values = self.lines[0].split()
        self.lines = lines[1:]
        if values[0] != 'pif':
            print "%s doesn't appear to be a PEST instruction file" % self.insflname
            return 0
        self.marker = values[1]
    @property
    def insflname(self):
        return self._insflname
    @insflname.setter
    def insflname(self,value):
        self._insflname = value
    @property
    def modelflname(self):
        return self._modelflname
    @modelflname.setter
    def modelflname(self,value):
        self._modelflname = value
    @property
    def marker(self):
        return self._marker
    @marker.setter
    def marker(self,value):
        self._marker = value 

class ModelTemplate(object):
    """pymads Template file class
    """
    def __init__(self,tplflname,modelflname):
        self.tplflname = tplflname
        self.modelflname = modelflname
        f = open( self.tplflname, 'r')
        self.lines = f.readlines()
        lines = array(self.lines)
        values = self.lines[0].split()
        self.lines = lines[1:]
        if values[0] != 'ptf':
            print "%s doesn't appear to be a PEST template file" % self.tplflname
            return 0
        self.marker = values[1]
    @property
    def tplflname(self):
        return self._tplflname
    @tplflname.setter
    def tplflname(self,value):
        self._tplflname = value
    @property
    def modelflname(self):
        return self._modelflname
    @modelflname.setter
    def modelflname(self,value):
        self._modelflname = value
    @property
    def marker(self):
        return self._marker
    @marker.setter
    def marker(self,value):
        self._marker = value 
 
def obj_fun(prob):
    of = 0.0
    for obsgrp in prob.obsgrp:
        for obs in obsgrp.observation:
            of += ( float(obs.value) - float(obs.sim_value) )**2
    return of
            
 
def main(argv=None):
    import sys
    if argv is None:
        argv = sys.argv
    pest_prob = read_pest(argv[1])
    print pest_prob

if __name__ == "__main__":
    main()
 
