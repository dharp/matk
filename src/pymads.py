from subprocess import call
from os import name
from pargrp import ParameterGroup
from obsgrp import ObservationGroup
import pesting
import calibrate
import forward

class PyMadsProblem(object):
    """ Problem class for pymads module
    """
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
        """ Number of model parameters
        """
        return self._npar
    @npar.setter
    def npar(self,value):
        self._npar = value
    @property
    def npargrp(self):
        """ Number of model parameter groups
        """
        return self._npargrp
    @npargrp.setter
    def npargrp(self,value):
        self._npargrp = value
    @property
    def nobs(self):
        """ Number of model observations
        """
        return self._nobs
    @nobs.setter
    def nobs(self,value):
        self._nobs = value
    @property
    def nobsgrp(self):
        """ Number of model observation groups
        """
        return self._nobsgrp
    @nobsgrp.setter
    def nobsgrp(self,value):
        self._nobsgrp = value 
    @property
    def ntplfile(self):
        """ Number of model template files
        """
        return self._ntplfile
    @ntplfile.setter
    def ntplfile(self,value):
        self._ntplfile = value       
    @property
    def ninsfile(self):
        """ Number of model instruction files
        """
        return self._ninsfile
    @ninsfile.setter
    def ninsfile(self,value):
        self._ninsfile = value       
    @property
    def sim_command(self):
        """ System command to run model
        """
        return self._sim_command
    @sim_command.setter
    def sim_command(self,value):
        self._sim_command = value
    def addpargrp(self, name, **kwargs):
        """Add a parameter group to the problem
        """
        self.pargrp.append(ParameterGroup(name, **kwargs))
    def addobsgrp(self, name):
        """Add a parameter group to the problem
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
        """ Add a template file to problem
        """
        self.tplfile.append(pesting.ModelTemplate(tplfilenm,model_infile))
    def addins(self,insfilenm,model_outfile):
        """ Add an instruction file to problem
        """
        self.insfile.append(pesting.ModelInstruction(insfilenm,model_outfile))
    def run_model(self):
        """ Run pymads problem forward model using current values
        """
        forward.run_model(self)
    def calibrate(self):
        """ Calibrate pymads problem model
        """
        x,cov_x,infodic,mesg,ier = calibrate.least_squares(self)
        return x,cov_x,infodic,mesg,ier
        
    