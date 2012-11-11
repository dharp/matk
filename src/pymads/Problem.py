""" Python module for MADS
"""
from pargrp import ParameterGroup
from obsgrp import ObservationGroup
import pesting

class Problem(object):
    def __init__(self, npar, nobs, ntplfile, ninsfile, **kwargs):
        self.npar = npar
        self.nobs = nobs
        self.ntplfile = ntplfile
        self.ninsfile = ninsfile
        self.npargrp = 1
        self.nobsgrp = 1
        self.sim_command = ''
        for k,v in kwargs.iteritems():
            if 'npargrp' == k:
                self.npargrp = v
            elif 'nobsgrp' == k:
                self.nobsgrp = v
            elif 'sim_command' == k:
                self.sim_command = v
            else:
                print k + ' is not a valid argument'
        self.pest = False
        self.pargrp = []
        self.obsgrp = []
        self.tplfile = []
        self.insfile = []
    @property
    def npar(self):
        return self._npar
    @npar.setter
    def npar(self,value):
        self._npar = value
    @property
    def npargrp(self):
        return self._npargrp
    @npargrp.setter
    def npargrp(self,value):
        self._npargrp = value
    @property
    def nobs(self):
        return self._nobs
    @nobs.setter
    def nobs(self,value):
        self._nobs = value
    @property
    def nobsgrp(self):
        return self._nobsgrp
    @nobsgrp.setter
    def nobsgrp(self,value):
        self._nobsgrp = value 
    @property
    def ntplfile(self):
        return self._ntplfile
    @ntplfile.setter
    def ntplfile(self,value):
        self._ntplfile = value       
    @property
    def ninsfile(self):
        return self._ninsfile
    @ninsfile.setter
    def ninsfile(self,value):
        self._ninsfile = value       
    @property
    def sim_command(self):
        """ Simulator command line string
        """
        return self._sim_command
    @sim_command.setter
    def sim_command(self,value):
        """ Simulator command line string
        """
        self._sim_command = value
    def addpargrp(self, name, **kwargs):
        """Add a parameter group to the problem
        
            [-] - optional parameters
            problem.addpargp( name, [
        """
        self.pargrp.append(ParameterGroup(name, **kwargs))
    def addobsgrp(self, name):
        """Add a parameter group to the problem
        
            [-] - optional parameters
            problem.addpargp( name, [
        """
        self.obsgrp.append(ObservationGroup(name))
    def __iter__(self):
        return self
    def next(self):
        if isinstance(self, ParameterGroup):
            if self.npargrp == 0:
                raise StopIteration
            index = self.npargrp - 1
            return self.pargrp[index]     
        elif isinstance(self, ObservationGroup):
            if self.nobsgrp == 0:
                raise StopIteration
            index = self.nobsgrp - 1
            return self.obsgrp[index]     
        elif isinstance(self, pesting.ModelTemplate):
            if self.ntplfile == 0:
                raise StopIteration
            index = self.ntplfile - 1
            return self.tplfile[index]     
        elif isinstance(self, pesting.ModelInstruction):
            if self.ninsfile == 0:
                raise StopIteration
            index = self.ninsfile - 1
            return self.insfile[index]      
    def addtpl(self,tplfilenm,model_infile):
        self.tplfile.append(pesting.ModelTemplate(tplfilenm,model_infile))
    def addins(self,insfilenm,model_outfile):
        self.insfile.append(pesting.ModelInstruction(insfilenm,model_outfile))
    