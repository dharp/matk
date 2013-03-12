__all__ = ['read_pest', 'read_pest_files', 'write_pest_files', 'ModelTemplate', 'ModelInstruction']

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
    pest_prob = pymads.PyMadsProblem(npar,nobs,ntplfile=ntplfile,ninsfile=ninsfile,npargrp=mynpargrp,
                        nobsgrp=mynobsgrp,pest=True)
    
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
        pest_prob.add_pargrp(pargrpnm,derinc=derinc,derinclb=derinclb,
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
        pest_prob.add_parameter(name,initial_value=initial_value,min=mn,max=mx,
                                trans=trans,scale=scale,offset=offset,
                                pargrpnm=pargrpnm,parchglim=parchglim)
            
    while '* ' not in f.readline():
        pass
    
    for i in range(pest_prob.nobsgrp):
        values = f.readline().split()
        obsgrpnm = values[0]
        pest_prob.add_obsgrp(obsgrpnm)
         
    if not '* ' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
  
    for i in range(pest_prob.nobs):
        values = f.readline().split()
        name = values[0]
        value = values[1]
        weight = values[2]
        obsgrpnm = values[3]
        pest_prob.add_observation(name,weight=weight,
                                  obsgrpnm=obsgrpnm,value=value)
    if not '* ' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
 
    pest_prob.sim_command = f.readline().strip()
    
    if not '* ' in f.readline():
        print "%s doesn't appear to be a PEST control file" % filename
        return 0
     
    for i in range(pest_prob.ntplfile):
        values = f.readline().split()
        pest_prob.add_tpl(values[0], values[1])
        
    for i in range(pest_prob.ninsfile):
        values = f.readline().split()
        pest_prob.add_ins(values[0], values[1])
 
    return pest_prob

def read_pest_files(prob, workdir=None):
    """ Collect simulated values from model files using
        pest instruction file

            Parameter
            ---------
            workdir : string
                name of directory where model output files exist            
    """
    for insfl in prob.insfile:
        line_index = -1
        if workdir:
            filename = workdir + '/' + insfl.modelflname
        else:
            filename = insfl.modelflname
        f = open( filename , 'r')
        model_file_lines = array(f.readlines())
        f.close()
        for line in insfl.lines:
            col_index = 0
            values = line.split()
            for val in values:
                if 'l' in val:
                    line_index += int(re.sub("l","", val))
                if 'w' in val:
                    col_index += 1
                if '!' in val:
                    obsnm = re.sub("!","", val)
                    values = model_file_lines[line_index].split()
                    prob.set_sim_value( obsnm, values[col_index])

def write_pest_files(prob, workdir=None):
    """ Write model from pest template file using current values

            Parameter
            ---------
            workdir : string
                name of directory to write model files to           
    """
    for tplfl in prob.tplfile:
        model_file_str = ''
        for line in tplfl.lines:
            model_file_str += line
        for par in prob.get_parameters():
            model_file_str = re.sub(tplfl.marker + r'.*' + par.name + r'.*' + tplfl.marker, 
                                        str(par.value), model_file_str)
        if workdir:
            filename = workdir + '/' + tplfl.modelflname
        else:
            filename = tplfl.modelflname
        f = open( filename, 'w')
        f.write(model_file_str)
        f.close()
        
class ModelInstruction(object):
    """pymads PEST instruction file class
    """
    def __init__(self,insflname,modelflname):
        self.insflname = insflname
        self.modelflname = modelflname
        f = open( self.insflname, 'r')
        self.lines = f.readlines()
        f.close()
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
        f.close()
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
 
