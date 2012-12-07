from pargrp import ParameterGroup
from obsgrp import ObservationGroup
import pesting
import calibrate
from run_model import *
from sample import *
from numpy import array,transpose

class PyMadsProblem(object):
    """ Problem class for pymads module
    """
    def __init__(self, npar, nobs, ntplfile, ninsfile, **kwargs):
        self.flag = {}
        self.npar = npar
        self.nobs = nobs
        self.ntplfile = ntplfile
        self.ninsfile = ninsfile
        self.npargrp = 1
        self.nobsgrp = 1
        self.sim_command = ''
        self.sample_size = 100
        self.ncpus = 1
        self.workdir_base = 'workdir'
        self.templatedir = None
        for k,v in kwargs.iteritems():
            if 'npargrp' == k:
                self.npargrp = v
            elif 'nobsgrp' == k:
                self.nobsgrp = v
            elif 'sim_command' == k:
                self.sim_command = v
            elif 'sample_size' == k:
                self.sample_size = int(v)
            elif 'ncpus' == k:
                self.ncpus == int(v)
            elif 'workdir_base' == k:
                self.workdir_base = v
            elif 'templatedir' == k:
                self.templatedir = v
            else:
                print k + ' is not a valid argument'
      
        self.pargrp = []
        self.obsgrp = []
        self.tplfile = []
        self.insfile = []
        self.flag['sims'] = False
        self.flag['obs'] = False
        self.flag['residual'] = False
        self.flag['pest'] = False
        # This flag is set to True on individual parallel runs in run_model.py
        self.flag['parallel'] = False 
        self.workdir_index = 0
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
    @property
    def sample_size(self):
        """ Set number of parameter samples to run
        """
        return self._sample_size
    @sample_size.setter
    def sample_size(self,value):
        self._sample_size = value
    @property
    def ncpus(self):
        """ Set number of cpus to use for concurrent model evaluations
        """
        return self._ncpus
    @ncpus.setter
    def ncpus(self,value):
        self._ncpus = value
    @property
    def workdir_base(self):
        """ Set the base name for parallel working directories
        """
        return self._workdir_base
    @workdir_base.setter
    def workdir_base(self,value):
        self._workdir_base = str(value)    
    @property
    def workdir_index(self):
        """ Set the working directory index for parallel runs    
        """
        return self._workdir_index
    @workdir_index.setter
    def workdir_index(self,value):
        self._workdir_index = value
    @property
    def templatedir(self):
        """ Set the name of the templatedir for parallel runs   
        """
        return self._templatedir
    @templatedir.setter
    def templatedir(self,value):
        self._templatedir = value
    def add_pargrp(self, name, **kwargs):
        """Add a parameter group to the problem
        """
        self.pargrp.append(ParameterGroup(name, **kwargs))
    def add_obsgrp(self, name):
        """Add a parameter group to the problem
        """
        self.obsgrp.append(ObservationGroup(name))
        self.flag['obs'] = True
    def get_observations(self):
        """ Get the observation values
        """
        if not self.flag['obs']:
            print 'Observations not defined in this problem'
            return 1
        obs = []
        for obsgrp in self.obsgrp:
            for observation in obsgrp.observation:
                    obs.append( observation )
        return obs
    def get_sim_values(self):
        """ Get the current simulated values
        """
        if not self.flag['sims']:
            print 'Simulated values do not exist, use run_model() first'
            return 1
        sims = []
        for obs in self.get_observations():
        #for obsgrp in self.obsgrp:
        #    for observation in obsgrp.observation:
            sims.append( obs.sim_value )
        return array( sims )
    def add_parameter(self, name, initial_value, **kwargs):
        """ Add parameter to problem
        """
        # Check if pargrpnm is identified, otherwise set to default
        pargrpnm = 'default'
        for k,v in kwargs.iteritems():
            if 'pargrpnm' == k:
                pargrpnm = str(v)
        # Check if pargrpnm has been added yet
        found = False
        for pgrp in self.pargrp:
            if pgrp.name == pargrpnm:
                found = True
                pgrp.add_parameter(name,initial_value,**kwargs)
        if not found:
            self.add_pargrp(pargrpnm)
            self.pargrp[-1].add_parameter(name,initial_value,**kwargs)
    def add_observation(self,name,value,**kwargs):
        """ Add observation to problem
        """
        # Check if pargrp is identified, otherwise set to default
        obsgrpnm = 'default'
        for k,v in kwargs.iteritems():
            if 'obsgrpnm' == k:
                obsgrpnm = str(v)
        # Check if obsgrpnm has been added yet
        found = False
        for ogrp in self.obsgrp:
            if ogrp.name == obsgrpnm:
                found = True
                ogrp.add_observation(name,value,**kwargs)
        if not found:
            self.add_obsgrp(obsgrpnm)
            self.obsgrp[-1].add_observation(name,value,**kwargs)
    def set_sim_value(self, obsnm, value):
        found = False
        for obsgrp in self.obsgrp:
            for obs in obsgrp.observation:
                if obs.name == obsnm:
                    obs.sim_value = value
                    found = True
        if not found:
            print "%s is not the name of an observation" % obsnm
            return 1
    def set_parameters(self,set_pars):
        """ Set parameters using values in first argument
        """
        index = 0
        for pargrp in self.pargrp:
            for par in pargrp.parameter:
                par.value = set_pars[index]
                index += 1 
    def get_parameters(self):
        """ Get array of parameter objects
        """
        pars = []
        for pargrp in self.pargrp:
            for par in pargrp.parameter:
                pars.append( par )
        return array( pars )
    def get_parameter_values(self):
        """ Get parameter values
        """
        values = []
        for par in self.get_parameters():
            values.append( par.value )
        return array( values )
    def get_parameter_names(self):
        """ Get parameter names
        """
        names = []
        for par in self.get_parameters():
            names.append( par.name )
        return array( names )
    def get_observation_values(self):
        """ Get observation values
        """
        names = []
        for obs in self.get_observations():
            names.append( obs.value )
        return array( names )
    def get_observation_names(self):
        """ Get observation names
        """
        names = []
        for obs in self.get_observations():
            names.append( obs.name )
        return array( names )
    def set_residuals(self):
        """ Get least squares values
        """
        if not self.flag['obs']:
            print 'Observations not defined in this problem'
            return 1
        if not self.flag['sims']:
            print 'Simulated values have not been generated, use run_model() first'
            return 1
        if self.flag['pest']:
            for obsgrp in self.obsgrp:
                for obs in obsgrp:
                    obs.residual = obs.value - obs.sim_value
        self.flag['residual'] = True
    def get_residuals(self):
        """ Get least squares values
        """
        if not self.flag['obs']:
            print 'Observations not defined in this problem'
            return 1
        if not self.flag['sims']:
            print 'Simulated values have not been generated, use run_model() first'
            return 1
        if self.flag['pest']:
            self.set_residuals()
        res = []
        for obsgrp in self.obsgrp:
            for obs in obsgrp:
                res.append(obs.residual)
        return array( res )
    def get_lower_bounds(self):
        """ Get parameter lower bounds
        """
        mini = []
        for pargrp in self.pargrp:
            for par in pargrp.parameter:
                mini.append(par.min)
        return array( mini )
    def get_upper_bounds(self):
        """ Get parameter lower bounds
        """
        maxi = []
        for pargrp in self.pargrp:
            for par in pargrp.parameter:
                maxi.append(par.max)
        return array( maxi )
    def get_dists(self):
        """ Get parameter probabilistic distributions
        """
        dists = []
        for pargrp in self.pargrp:
            for par in pargrp.parameter:
                dists.append(par.dist)
        return array( dists )
    def get_dist_pars(self):
        """ Get parameters needed by parameter distributions
        """
        dist_pars = []
        for pargrp in self.pargrp:
            for par in pargrp.parameter:
                dist_pars.append(par.dist_pars)
        return array( dist_pars )
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
    def add_tpl(self,tplfilenm,model_infile):
        """ Add a template file to problem
        """
        self.tplfile.append(pesting.ModelTemplate(tplfilenm,model_infile))
    def add_ins(self,insfilenm,model_outfile):
        """ Add an instruction file to problem
        """
        self.insfile.append(pesting.ModelInstruction(insfilenm,model_outfile))
    def run_model(self):
        """ Run pymads problem forward model using current values
        """
        forward(self)
        self.flag['sims'] = True
    def run_parallel(self):
        """ Run models concurrently on multiprocessor machine
        """
        if not self.flag['parallel']:
            print 'Parallel execution not enable, set ncpus to number of processors'
            return 0
        #run_model.parallel(self)
    def calibrate(self):
        """ Calibrate pymads problem model
        """
        x,cov_x,infodic,mesg,ier = calibrate.least_squares(self)
        return x,cov_x,infodic,mesg,ier
    def get_samples(self, siz=100, noCorrRestr=False, corrmat=None, outfile=None, seed=None):
        """ Draw lhs samples from scipy.stats module distribution
        
            Parameter
            ---------
            siz : int
                number of samples to generate, ignored if samples are provided
            noCorrRestr: bool
                if True, correlation structure is not enforced on sample
            corrmat : matrix
                correlation matrix
            outfile : string
                name of file to output samples to
            seed : int
                random seed to allow replication of samples
            
            Returns
            -------
            samples : ndarray 
                Parameter samples
            outfile : string
                name of file to write samples in.
                If outfile=None, no file is written.
          
        """
        # If siz specified, set sample_size
        if siz:
            self.sample_size = siz
        x = get_samples(self,siz=self.sample_size, noCorrRestr=noCorrRestr,
                         corrmat=corrmat, seed=seed)
        x =  array(x).transpose()
        if outfile:
            f = open(outfile, 'w')
            f.write( '%-9s '%'id ' )
            for parnm in self.get_parameter_names():
                f.write( '%22s '%parnm)
            f.write( '\n')
            for sid in range(siz):
                f.write( '%-9d '%(int(sid) + 1))
                for val in x[sid]:
                    f.write( '%22.16e '% val)
                f.write( '\n')
            f.close() 
        return x
    def run_samples(self, siz=100, noCorrRestr=False, corrmat=None,
                     samples=None, outfile=None, parallel=False, ncpus=None,
                      templatedir=None, workdir_base=None, seed=None):
        """ Use or generate samples and run models
            First argument (optional) is an array of samples
            
            Parameter
            ---------
            siz : int
                number of samples to generate, ignored if samples are provided
            noCorrRestr: bool
                if True, correlation structure is not enforced on sample
            corrmat : matrix
                correlation matrix
            samples : ndarray
                matrix of samples, npar columns by siz rows
            outfile : string
                name of file to write samples and responses in. 
                If outfile=None, no file is written.
            parallel : bool
                if True, models run concurrently with 'ncpus' cpus
            ncpus : int
                number of cpus to use to run models concurrently
            templatedir : string
                name of folder including files needed to run model
                (e.g. template files, instruction files, executables, etc.)
            workdir_base : string
                base name for model run folders, run index is appended
                to workdir_base
            seed : int
                random seed to allow replication of samples
            
            Returns
            -------
            responses : ndarray 
                Responses from model runs
            samples : ndarray 
                Parameter samples, same as input samples if provided
            
        """
        responses, samples = run_samples(self, siz=siz, samples=samples,
                 noCorrRestr=noCorrRestr, corrmat=corrmat,outfile=outfile, 
                 parallel=parallel, ncpus=ncpus, templatedir=templatedir,
                workdir_base=workdir_base, seed=seed)
        if outfile:
            f = open(outfile, 'w')
            f.write( '%-9s '%'id ' )
            for parnm in self.get_parameter_names():
                f.write( '%22s '%parnm)
            for obsnm in self.get_observation_names():
                f.write( '%22s '%obsnm)
            f.write( '\n')
            for sid in range(siz):
                f.write( '%-9d '%(int(sid) + 1))
                for val in samples[sid]:
                    f.write( '%22.16e '% val)
                for val in responses[sid]:
                    f.write( '%22.16e '% val)
                f.write( '\n')
            f.close()
        return responses, samples
    def parallel(self, ncpus, par_sets, templatedir=None, workdir_base=None ):
        responses, samples, status = parallel(self, ncpus, par_sets, templatedir=templatedir,
                            workdir_base=workdir_base)
        if status:
            return 0, 0
        else:
            return responses, samples
    