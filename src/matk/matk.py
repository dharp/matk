import sys, os
import pdb
from parameter import Parameter
from observation import Observation
from sampleset import SampleSet
import numpy 
from lhs import *
import cPickle as pickle
from shutil import rmtree
import itertools
from multiprocessing import Process, Manager, Pool, freeze_support
from multiprocessing.queues import Queue, JoinableQueue
import traceback
from copy import deepcopy
import pest_io
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from lmfit.asteval import Interpreter
import warnings
from scipy.stats import rv_discrete

class matk(object):
    """ Class for Model Analysis ToolKit (MATK) module
    """
    def __init__(self, model='', model_args=None, model_kwargs={}, cpus=1,
                 workdir_base=None, workdir=None, results_file=None,
                 seed=None, sample_size=10, hosts={}):
        '''Initialize MATK object
        :param model: Python function whose first argument is a dictionary of parameters and returns model outputs
        :type model: str
        :param model_args: additional arguments to model
        :type model_args: any
        :param model_kwargs: additional keyword arguments to model
        :type model_kwargs: any
        :param cpus: number of cpus to use
        :type cpus: int
        :param workdir_base: Base name of directory to use for model runs (parallel run case), run numbers are appended to base name
        :type workdir_base: str
        :param workdir: Name of directory to use for model runs (serial run case)
        :type workdir: str
        :param results_file: Name of file to write results
        :type results_file: str
        :param seed: Seed for random number generator
        :type seed: int
        :param sample_size: Size of sample to generate
        :type sample_size: int
        :param hosts: Host names to run on (i.e. on a cluster), hostname provided as kwarg to model (hostname=<hostname>)
        :type hosts: lst(str)
        :returns: object -- MATK object
        '''
        self.model = model
        self._model_args = model_args
        self._model_kwargs = model_kwargs
        self.cpus = cpus
        self.workdir_base = workdir_base
        self.workdir = workdir
        self.results_file = results_file
        self.seed = seed
        self.sample_size = sample_size
        self.hosts = hosts
      
        self.pars = OrderedDict()
        self.discrete_pars = OrderedDict()
        self.obs = OrderedDict()
        self.sampleset = OrderedDict()
        self.workdir_index = 0
        self._current = False # Flag indicating if simulated values are associated with current parameters
    #def __repr__(self):
    #    s = 'MATK Model Analysis Object\n\n'
    #    s += 'Model: '+self.model.func_name+'\n\n'
    #    s += 'Number of Parameters: '+str(len(self.pars))+'\n'
    #    if len(self.pars) < 11:
    #        s+='Parameters:\n'
    #        for k,v in self.pars.iteritems(): s += repr(v); s+='\n'
    #    s += '\nNumber of Observations: '+str(len(self.obs))+'\n'
    #    if len(self.obs) < 11:
    #        s+='Observations:\n'
    #        for k,v in self.obs.iteritems(): s += repr(v); s+='\n'
    #    else:
    #        s+='Too many observations to display\n'
    #        s+='Use "obs" attribute to return observation dictionary\n'
    #    return s
    @property
    def model(self):
        """ Python function that runs model
        """
        return self._model
    @model.setter
    def model(self,value):
        self._model = value       
    @property
    def model_args(self):
        """ Tuple of extra arguments to MATK model expected to come after parameter dictionary
        """
        return self._model_args
    @model_args.setter
    def model_args(self,value):
        if value is None:
            self._model_args = value
        elif not isinstance( value, (tuple,list,numpy.ndarray) ):
            print "Error: Expected list or array for model keyword arguments"
            return
        else:
            self._model_args = value
    @property
    def model_kwargs(self):
        """ Dictionary of extra keyword arguments to MATK model expected to come after parameter dictionary and model_args
        """
        return self._model_kwargs
    @model_kwargs.setter
    def model_kwargs(self,value):
        if value is None:
            self._model_kwargs = value       
        elif not isinstance( value, dict ):
            print "Error: Expected dictionary for model keyword arguments"
            return
        else:
            self._model_kwargs = value       
    @property
    def cpus(self):
        """ Set number of cpus to use for concurrent model evaluations
        """
        return self._cpus
    @cpus.setter
    def cpus(self,value):
        self._cpus = value
    @property
    def workdir_base(self):
        """ Set the base name for parallel working directories
        """
        return self._workdir_base
    @workdir_base.setter
    def workdir_base(self,value):
        self._workdir_base = value    
    @property
    def workdir(self):
        """ Set the base name for parallel working directories
        """
        return self._workdir
    @workdir.setter
    def workdir(self,value):
        self._workdir = value    
    @property
    def workdir_index(self):
        """ Set the working directory index for parallel runs    
        """
        return self._workdir_index
    @workdir_index.setter
    def workdir_index(self,value):
        self._workdir_index = value
    @property
    def results_file(self):
        """ Set the name of the results_file for parallel runs   
        """
        return self._results_file
    @results_file.setter
    def results_file(self,value):
        self._results_file = value
    @property
    def seed(self):
        """ Set the seed for random sampling
        """
        return self._seed
    @seed.setter
    def seed(self,value):
        self._seed = value
    def ssr(self,group=None):
        """ Sum of squared residuals

            :param group: Group name of observations; if not None, ssr for observation group will be returned
            :type group: str
        """
        return sum(numpy.array(self.residuals(group))**2)
    def add_par(self, name, value=None, vary=True, min=None, max=None, expr=None, discrete_vals=[], **kwargs):
        """ Add parameter to problem

            :param name: Name of parameter
            :type name: str
            :param value: Initial parameter value
            :type value: float
            :param vary: Whether parameter should be varied or not, currently only used with lmfit
            :type vary: bool
            :param min: Minimum bound
            :type min: float
            :param max: Maximum bound
            :type max: float
            :param expr: Mathematical expression to use to calculate parameter value
            :type expr: str
            :param discrete_vals: tuple of two array_like defining discrete values and associated probabilities
            :type discrete_vals: (lst,lst)
            :param kwargs: keyword arguments passed to parameter class
        """
        if name in self.pars: 
            self.pars[name] = Parameter(name,parent=self,value=value,vary=vary,min=min,max=max,expr=expr,discrete_vals=discrete_vals,**kwargs)
        else:
            self.pars.__setitem__( name, Parameter(name,parent=self,value=value,vary=vary,min=min,max=max,expr=expr,discrete_vals=discrete_vals,**kwargs))
    def add_obs(self,name, sim=None, weight=1.0, value=None, group=None):
        ''' Add observation to problem
            
            :param name: Observation name
            :type name: str
            :param sim: Simulated value
            :type sim: fl64
            :param weight: Observation weight
            :type weight: fl64
            :param value: Value of observation
            :type value: fl64
            :param group: Name identifying group or type of observation
            :type group: str
            :returns: Observation object
        '''
        if name in self.obs: 
            self.obs[name] = Observation(name,sim=sim,weight=weight,value=value,group=group)
        else:
            self.obs.__setitem__( name, Observation(name,sim=sim,weight=weight,value=value,group=group))
    def create_sampleset(self,samples,name=None,responses=None,indices=None,index_start=1):
        """ Add sample set to problem
            
            :param name: Name of sample set
            :type name: str
            :param samples: Matrix of parameter samples with npar columns in order of matk.pars.keys()
            :type samples: list(fl64),ndarray(fl64)
            :param responses: Matrix of associated responses with nobs columns in order matk.obs.keys() if observation exists (existence of observations is not required) 
            :type responses: list(fl64),ndarray(fl64)
            :param indices: Sample indices to use when creating working directories and output files
            :type indices: list(int),ndarray(int)
        """
        if not isinstance( samples, (list,numpy.ndarray)):
            print "Error: Parameter samples are not a list or ndarray"
            return 1
        npar = len(self.pars)
        # If list, convert to ndarray
        if isinstance( samples, list ):
            samples = numpy.array(samples)
        if not samples.shape[1] == npar:
            print "Error: The number of columns in sample is not equal to the number of parameters in the problem"
            return 1
        if name is None:
            ind = str(len(self.sampleset))
            name = 'ss'+str(ind)
            while name in self.sampleset:
                ind += 1
                name = 'ss'+str(ind)
        if len(self.pars) > 0:
            parnames = self.parnames
        else:
            parnames = None
        if len(self.obs) > 0:
            obsnames = self.obsnames
        else:
            obsnames = None
        if name in self.sampleset: 
            self.sampleset[name] = SampleSet(name,samples,parent=self,responses=responses,
                                             indices=indices,index_start=index_start)
        else:
            self.sampleset.__setitem__( name, SampleSet(name,samples,parent=self,responses=responses,
                                                indices=indices,index_start=index_start))
        return self.sampleset[name]
    def read_sampleset(self, file, name=None):
        """ Read MATK output file and assemble corresponding sampleset with responses.
        
        :param name: Name of sample set
        :type name: str
        :param file: Path to MATK output file
        :type file: str
        """
        # open file
        if not os.path.isfile(file):
            print 'No file '+file+' found...'
            return
        fp = open(file)
        # parse file
        npar = int(fp.readline().rstrip().split(':')[1])
        nobs = int(fp.readline().rstrip().split(':')[1])
        headers = fp.readline().rstrip().split()
        data = numpy.array([[float(num) for num in line.split()] for line in fp if not line.isspace()])
        indices = numpy.array([int(v) for v in data[:,0]])
        # add parameters
        for header,dat in zip(headers[1:npar+1],data[:,1:npar+1].T):
            if header not in self.pars:
                self.add_par(header,min = numpy.min(dat),max = numpy.max(dat))
        # add observations
        for header in headers[npar+1:]: 
            if header not in self.obs:
                self.add_obs(header)
        # create samples
        samples = data[:,1:npar+1]
        if nobs > 0:
            responses = data[:,npar+1:]
        else: responses = None
        return self.create_sampleset(samples,name=name,responses=responses,indices=indices)
    def copy_sampleset(self,oldname,newname=None):
        """ Copy sampleset

            :param oldname: Name of sampleset to copy
            :type oldname: str
            :param newname: Name of new sampleset
            :type newname: str
        """
        return self.create_sampleset(self.sampleset[oldname].samples.values,name=newname,responses=self.sampleset[oldname].responses.values,indices=self.sampleset[oldname].indices)
    @property
    def simvalues(self):
        """ Simulated values
            :returns: lst(fl64) -- simulated values in order of matk.obs.keys()
        """
        return numpy.array([obs.sim for obs in self.obs.values()])
    @property
    def simdict(self):
        """ Simulated values
            :returns: lst(fl64) -- simulated values in order of matk.obs.keys()
        """
        return dict(zip(self.obsnames,self.simvalues))
    def _set_simvalues(self, *args, **kwargs):
        """ Set simulated values using a tuple, list, numpy.ndarray, dictionary or keyword arguments
        """
        if len(args) > 0 and len(kwargs) > 0:
            print "Warning: dictionary arg will overide keyword args"
        if len(args) > 0:
            if isinstance( args[0], dict ):
                for k,v in args[0].iteritems():
                    if k in self.obs:
                        self.obs[k].sim = v
                    else:
                        self.add_obs( k, sim=v ) 
            elif isinstance( args[0], (list,tuple,numpy.ndarray) ):
                # If no observations exist, create them
                if len(self.obs) == 0:
                    for i,v in zip(range(len(args[0])),args[0]): 
                        self.add_obs('obs'+str(i+1),sim=v)
                elif not len(args[0]) == len(self.obs): 
                    print len(args[0]), len(self.obs)
                    print "Error: Number of simulated values in list or tuple does not match created observations"
                    return
                else:
                    for k,v in zip(self.obs.keys(),args[0]):
                        self.obs[k].sim = v
        else:
            for k,v in kwargs.iteritems():
                if k in self.obs:
                    self.obs[k].sim = v
                else:
                    self.add_obs( k, sim=v ) 
    @property
    def parvalues(self):
        """ Parameter values
        """
        return numpy.array([par._val for par in self.pars.values()])
    @parvalues.setter
    def parvalues(self, value):
        """ Set parameter values using a tuple, list, numpy.ndarray, or dictionary
        """
        if isinstance( value, dict ):
            for k,v in value.iteritems():
                self.pars[k]._val = v
        elif isinstance( value, (list,tuple,numpy.ndarray)):
            if not len(value) == len(self.pars): 
                print "Error: Number of parameter values in ndarray does not match created parameters"
                return
            for k,v in zip(self.parnames,value):
                self.pars[k]._val = v
        else:
            print "Error: tuple, list, numpy.ndarray, or dictionary expected"
    @property
    def parnames(self):
        """ Get parameter names
        """
        return [par.name for par in self.pars.values()]
    @property
    def obsvalues(self):
        """ Observation values
        """
        return numpy.array([o.value for o in self.obs.values()])
    @obsvalues.setter
    def obsvalues(self, value):
        """ Set observed values using a tuple, list, numpy.ndarray, or dictionary
        """
        if isinstance( value, dict ):
            for k,v in value.iteritems():
                if k in self.obs:
                    self.obs[k].value = v
                else:
                    self.add_obs( k, value=v ) 
        elif isinstance( value, (list,tuple,numpy.ndarray) ):
            # If no observations exist, create them
            if len(self.obs) == 0:
                for i,v in enumerate(value): 
                    self.add_obs('obs'+str(i),value=v)
            # else, check if length of value is equal to number created observation
            elif not len(value) == len(self.obs): 
                    print "Error: Number of simulated values does not match created observations"
                    return
            # else, set observation values in order
            else:
                for k,v in zip(self.obs.keys(),value):
                    self.obs[k].value = v
        else:
            print "Error: tuple, list, numpy.ndarray, or dictionary expected"
    @property
    def obsnames(self):
        """ Get observation names
        """
        return [o.name for o in self.obs.values()]
    @property
    def obsweights(self):
        """ Get observation weights
        """
        return numpy.array([o.weight for o in self.obs.values()])
    @property
    def obsgroups(self):
        """ Get observation groups
        """
        return numpy.array([o.group for o in self.obs.values()])
    def residuals(self,group=None):
        """ Get least squares values

            :param group: Group name of observations; if not None, only residuals in group will be returned
            :type group: str
        """
        if group is None:
            return numpy.array([o.residual for o in self.obs.values()])
        else:
            return numpy.array([o.residual for o in self.obs.values() if o.group == group])
    @property
    def parmins(self):
        """ Get parameter lower bounds
        """
        return numpy.array([par.min for par in self.pars.values()])
    @property
    def parmaxs(self):
        """ Get parameter upper bounds
        """
        return numpy.array([par.max for par in self.pars.values()])
    @property
    def pardists(self):
        """ Get parameter probabilistic distributions
        """
        return [par.dist for par in self.pars.values()]
    @property
    def pardist_pars(self):
        """ Get parameters needed by parameter distributions
        """
        return numpy.array([par.dist_pars for par in self.pars.values()])
    @property
    def nomvalues(self):
        """ Nominal parameter values used in info gap analyses
        """
        return numpy.array([par.nominal for par in self.pars.values()])
    @nomvalues.setter
    def nomvalues(self, value):
        """ Set nominal parameter values using a tuple, list, numpy.ndarray, or dictionary
        """
        if isinstance( value, dict ):
            for k,v in value.iteritems():
                self.pars[k].nominal = v
        elif isinstance( value, (list,tuple,numpy.ndarray)):
            if not len(value) == len(self.pars): 
                print "Error: Number of parameter values in ndarray does not match created parameters"
                return
            for v,k in zip(value,self.pars.keys()):
                self.pars[k].nominal = v
        else:
            print "Error: tuple, list, numpy.ndarray, or dictionary expected"
    def __iter__(self):
        return self
    def make_workdir(self, workdir=None, reuse_dirs=False):
        """ Create a working directory

            :param workdir: Name of directory where model will be run. It will be created if it does not exist
            :type workdir: str
            :param reuse_dirs: If True and workdir exists, the model will reuse the directory
            :type reuse_dirs: bool
            :returns: int -- 0: Successful run, 1: workdir exists 
        """
        if not workdir is None: self.workdir = workdir
        if not self.workdir is None:
            # If folder doesn't exist
            if not os.path.isdir( self.workdir ):
                os.makedirs( self.workdir )
                return 0
            # or if reusing directories
            elif reuse_dirs:
                pass
                return 0
            # or throw error
            else:
                print "Error: " + self.workdir + " already exists"
                return 1
    def forward(self, pardict=None, workdir=None, reuse_dirs=False, job_number=None, hostname=None, processor=None):
        """ Run MATK model using current values

            :param pardict: Dictionary of parameter values keyed by parameter names
            :type pardict: dict
            :param workdir: Name of directory where model will be run. It will be created if it does not exist
            :type workdir: str
            :param reuse_dirs: If True and workdir exists, the model will reuse the directory
            :type reuse_dirs: bool
            :param job_number: Sample id
            :type job_number: int
            :param hostname: Name of host to run job on, will be passed to MATK model as kwarg 'hostname'
            :type hostname: str
            :param processor: Processor id to run job on, will be passed to MATK model as kwarg 'processor'
            :type processor: str or int
            :returns: int -- 0: Successful run, 1: workdir exists 
        """
        if not workdir is None: self.workdir = workdir
        if not self.workdir is None:
            curdir = os.getcwd()
            status = self.make_workdir( workdir=self.workdir, reuse_dirs=reuse_dirs)
            os.chdir( self.workdir )
            if status:
                return 1
        else:
            curdir = None
			
        # Set job_number if among the model keyword arguments
        if 'job_number' in self.model_kwargs:
            self.model_kwargs['job_number'] = job_number

        if hasattr( self.model, '__call__' ):
            try:
                if pardict is None:
                    pardict = dict([(k,par.value) for k,par in self.pars.items()])
                else:
                    self.parvalues = pardict
                    pardict = dict(zip(self.parnames,self.parvalues))
                if self.model_args is None and not self.model_kwargs:
                    if hostname is None: sims = self.model( pardict )
                    else:
                        if processor is None: sims = self.model( pardict, hostname=hostname )
                        else: sims = self.model( pardict, hostname=hostname, processor=processor )
                elif not self.model_args is None and not self.model_kwargs:
                    if hostname is None: sims = self.model( pardict, *self.model_args )
                    else:
                        if processor is None: sims = self.model( pardict, *self.model_args, hostname=hostname )
                        else: sims = self.model( pardict, *self.model_args, hostname=hostname, processor=processor )
                elif self.model_args is None and self.model_kwargs:
                    if hostname is None: sims = self.model( pardict, **self.model_kwargs )
                    else:
                        if processor is None: sims = self.model( pardict, hostname=hostname, **self.model_kwargs )
                        else: sims = self.model( pardict, hostname=hostname, processor=processor, **self.model_kwargs )
                elif not self.model_args is None and self.model_kwargs:
                    if hostname is None: sims = self.model( pardict, *self.model_args, **self.model_kwargs )
                    else:
                        if processor is None: sims = self.model( pardict, *self.model_args, hostname=hostname, **self.model_kwargs )
                        else: sims = self.model( pardict, *self.model_args, hostname=hostname, processor=processor, **self.model_kwargs )
                self._current = True
                if not curdir is None: os.chdir( curdir )
                if sims is not None:
                    if isinstance(sims,(float,int)): sims = [sims]
                    # Remove extra sims items if not in current observations
                    if isinstance(sims,(dict,OrderedDict)) and len(self.obs) > 0: 
                        sims = OrderedDict([(k,v) for k,v in sims.iteritems() if k in self.obs])
                    if len(sims):
                        self._set_simvalues(sims)
                        simdict = OrderedDict(zip(self.obsnames,self.simvalues))
                        return simdict
                else: return None
            except:
                errstr = traceback.format_exc()                
                if not curdir is None: os.chdir( curdir )
                s = "-"*60+'\n'
                if job_number is not None:
                    s += "Exception in job "+str(job_number)+":\n"
                else:
                    s += "Exception in model call:\n"
                s += errstr
                s += "-"*60
                print s
                return s
        else:
            print "Error: Model is not a Python function"
            if not curdir is None: os.chdir( curdir )
            return 1
    def lmfit(self,maxfev=0,report_fit=True,cpus=1,epsfcn=None,xtol=1.e-7,ftol=1.e-7,
              workdir=None, verbose=False, save_evals=False, difference_type='forward',
              **kwargs):
        """ Calibrate MATK model using lmfit package

            :param maxfev: Max number of function evaluations, if 0, 100*(npars+1) will be used
            :type maxfev: int
            :param report_fit: If True, parameter statistics and correlations are printed to the screen
            :type report_fit: bool
            :param cpus: Number of cpus to use for concurrent simulations during jacobian approximation
            :type cpus: int
            :param epsfcn: jacobian finite difference approximation increment (single float of list of npar floats)
            :type epsfcn: float or lst[float]
            :param xtol: Relative error in approximate solution
            :type xtol: float
            :param ftol: Relative error in the desired sum of squares
            :type ftol: float
            :param workdir: Name of directory to use for model runs, calibrated parameters will be run there after calibration 
            :type workdir: str
            :param verbose: If true, print diagnostic information to the screen
            :type verbose: bool
            :param difference_type: Type of finite difference approximation, 'forward' or 'central'
            :type difference_type: str
            :param save_evals: If True, a MATK sampleset of calibration function evaluation parameters and responses will be returned
            :type save_evals: bool
            :returns: tuple(lmfit minimizer object; parameter object; if save_evals=True, also returns a MATK sampleset of calibration function evaluation parameters and responses)

            Additional keyword argments will be passed to scipy leastsq function:
            http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.leastsq.html
        """
           
        try: import lmfit
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import lmfit module. ({})".format(exc))
            return
        self.cpus = cpus
        if save_evals: self._minimize_pars = []; self._minimize_sims = []

        # Create lmfit parameter object
        params = lmfit.Parameters()
        for k,p in self.pars.items():
            params.add(k,value=p.value,vary=p.vary,min=p.min,max=p.max,expr=p.expr) 

        out = lmfit.minimize(self.__lmfit_residual, params, args=(cpus,epsfcn,workdir,verbose,save_evals,difference_type), 
                maxfev=maxfev,xtol=xtol,ftol=ftol,Dfun=self.__jacobian, **kwargs)

        # Make sure that self.pars are set to final values of params
        nm = [params[k].name for k in self.pars.keys()]
        vs = [params[k].value for k in self.pars.keys()]
        self.parvalues = dict(zip(nm,vs))
        # Run forward model to set simulated values
        if isinstance( cpus, int):
            self.forward(workdir=workdir,reuse_dirs=True)
        elif isinstance( cpus, dict):
            hostname = cpus.keys()[0]
            processor = cpus[hostname][0]
            self.forward(workdir=workdir,reuse_dirs=True,
                         hostname=hostname,processor=processor)
        else:
            print 'Error: cpus argument type not recognized'
            return

        if report_fit:
            print lmfit.report_fit(params)
            print 'SSR: ',self.ssr()
        if save_evals:
            return out, params, self.create_sampleset(self._minimize_pars, responses=self._minimize_sims)
        else:
            return out,params
    def __lmfit_residual(self, params, cpus=1, epsfcn=None, workdir=None,verbose=False,save_evals=False,difference_type='forward'):
        if verbose: print 'forward run: ',params
        pardict = dict([(k,n.value) for k,n in params.items()])
        if isinstance( cpus, int):
            self.forward(pardict=pardict,workdir=workdir,reuse_dirs=True)
        elif isinstance( cpus, dict):
            hostname = cpus.keys()[0]
            processor = cpus[hostname][0]
            self.forward(pardict=pardict,workdir=workdir,reuse_dirs=True,
                         hostname=hostname,processor=processor)
        else:
            print 'Error: cpus argument type not recognized'
            return
        if verbose: 
            if len(numpy.unique(self.obsgroups)) > 1:
                for grp in numpy.unique(self.obsgroups):
                    if grp is not None:
                        print '{} SSR: {}'.format( grp, self.ssr(grp))
                print 'Total SSR: ', self.ssr()
            else:
                print 'SSR: ', self.ssr()
        if save_evals:
            self._minimize_pars.append(self.parvalues)
            self._minimize_sims.append(self.simvalues)
        return self.residuals()
    def __jacobian( self, params, cpus=1, epsfcn=None, workdir_base=None,verbose=False,save=False,
                   difference_type='forward',reuse_dirs=True):
        ''' Numerical Jacobian calculation
        '''
        # Collect parameter values
        a = numpy.array([k.value for k in params.values()])
        # Collect array of 1s and 0s to indicate variable or fixed parameters
        vary = numpy.array([int(k.vary) for k in params.values()])
        # Determine finite difference increment for each parameter
        if epsfcn is None:
            hs = numpy.sqrt(numpy.finfo(float).eps)*a
            hs[numpy.where(hs==0)[0]] = numpy.sqrt(numpy.finfo(float).eps)
        elif isinstance(epsfcn,float):
            hs = epsfcn * numpy.ones(len(a))
        else:
            if len(epsfcn) == len(a):
                hs = numpy.array(epsfcn)
            elif len(epsfcn) == numpy.sum(vary):
                hs = []
                i = 0
                for v in vary: 
                    if v: hs.append(epsfcn[i])
                    else: hs.append(0.)
            else:
                print "\nError: length of epsfcn array is not the number of parameters or number of free (vary=True) parameters\n"
                return 1
        # Make fixed hs values zero
        hs = hs*vary
        # Forward differences
        humat = numpy.identity(len(a))*hs
        # Remove zero rows associated with fixed parameters
        humat = humat[~numpy.all(humat == 0, axis=1)]
        parset = [a]*humat.shape[0] + humat
        if difference_type == 'central':
            parset = numpy.append(parset, [a]*humat.shape[0] - humat, axis=0)
        elif difference_type == 'forward':
            parset = numpy.append(parset,[a],axis=0)
        else:
            print 'difference_type not recognized, expects "forward" or "central"'
            return
        if verbose: 
            print "Jacobian parameter combinations:"
            numpy.set_printoptions(precision=16)
            print parset
        if cpus > 1:
            self.create_sampleset(parset,name='_jac_')

            # Perform simulations on parameter sets
            self.sampleset['_jac_'].run( cpus=cpus, verbose=False,
                             workdir_base=workdir_base, save=False, reuse_dirs=reuse_dirs )
            sims = self.sampleset['_jac_'].responses.values
            if verbose and len(self.obs):
                print "Jacobian sse's:"
                print self.sampleset['_jac_'].sse()
        else:
            sims = []
            if verbose and len(self.obs): sse = []
            for ps in parset:
                self.forward(pardict=dict(zip(self.parnames,ps)),workdir=workdir_base,reuse_dirs=True)
                if verbose and len(self.obs): sse.append(self.ssr())
                sims.append(self.simvalues)
            if verbose and len(self.obs):
                print "Jacobian sse's:"
                print sse
        J = []
        # Delete hs's associated with fixed parameters
        #hs = numpy.delete(hs,numpy.where(hs==0))
        if difference_type == 'central':
            fsims = sims[:numpy.sum(vary)]
            bsims = sims[numpy.sum(vary):]
            for h,fsim,bsim in zip(hs[numpy.where(vary==1)[0]],fsims,bsims):
                J.append((bsim-fsim)*self.obsweights/(2*h))
        elif difference_type == 'forward':
            diffsims = sims[:numpy.sum(vary)]
            zerosims = sims[-1]
            for h,d in zip(hs[numpy.where(vary==1)[0]],diffsims):
                J.append((zerosims-d)*self.obsweights/h)
        self.parvalues = a
        return numpy.array(J).T

    def minimize(self,method='SLSQP',maxiter=100,workdir=None,bounds=(),constraints=(),options={'eps':1.e-3},save_evals=False):
        """ Minimize a scalar function of one or more variables

            :param maxiter: Max number of iterations
            :type maxiter: int
            :param workdir: Name of directory to use for model runs, calibrated parameters will be run there after calibration 
            :type workdir: str
            :returns: OptimizeResult; if save_evals=True, also returns a MATK sampleset of calibration function evaluation parameters and responses
        """
        try: from scipy.optimize import minimize
        except ImportError as exc:
            sys.stderr.write("Error: failed to import scipy.optimize.minimize module. ({})".format(exc))
            return
        if save_evals: self._minimize_pars = []; self._minimize_sims = []
        if len(bounds) == 0:
            bounds = zip(self.parmins,self.parmaxs)
        x0 = self.parvalues
        res = minimize(self.__minimize_residual,x0,args=(workdir,save_evals),method=method,bounds=bounds,constraints=constraints,options=options)
        if save_evals:
            return res, self.create_sampleset(self._minimize_pars, responses=self._minimize_sims)
        else: return res

    def __minimize_residual(self, x, workdir, save_evals):
        pardict = dict([(k,n) for k,n in zip(self.parnames,x)])
        self.forward(pardict=pardict,workdir=workdir,reuse_dirs=True)
        if save_evals:
            self._minimize_pars.append(self.parvalues)
            self._minimize_sims.append(self.simvalues)
        return numpy.abs(self.residuals()[0])

    def levmar(self,workdir=None,reuse_dirs=False,max_iter=1000,full_output=True):
        """ Calibrate MATK model using levmar package

            :param workdir: Name of directory where model will be run. It will be created if it does not exist
            :type workdir: str
            :param reuse_dirs: If True and workdir exists, the model will reuse the directory
            :type reuse_dirs: bool
            :param max_iter: Maximum number of iterations
            :type max_iter: int
            :param full_output: If True, additional output displayed during calibration
            :returns: levmar output
        """
        try: import levmar
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import levmar module. ({})".format(exc))
            return
        def _f(pars, prob):
            prob = prob[0]
            nm = [p.name for p in prob.pars.values()]
            vs = [p._func_value(v) for v,p in zip(pars,prob.pars.values())]
            print nm,vs
            prob.forward(pardict=dict(zip(nm,vs)),workdir=workdir,reuse_dirs=reuse_dirs)
            return prob.simvalues
        vs = [p.from_internal for p in self.pars.values()]
        meas = self.obsvalues
        if full_output: full_output = 1
        out = levmar.leastsq(_f, vs, meas, args=(self,), Dfun=None, max_iter=max_iter, full_output=full_output)
        #TODO Put levmar results into MATK object
        return out
    def lhs(self, name=None, siz=None, noCorrRestr=False, corrmat=None, seed=None, index_start=1):
        """ Draw lhs samples of parameter values from scipy.stats module distribution
        
            :param name: Name of sample set to be created
            :type name: str
            :param siz: Number of samples to generate, ignored if samples are provided
            :type siz: int
            :param noCorrRestr: If True, correlation structure is not enforced on sample, use if siz is less than number of parameters
            :type noCorrRestr: bool
            :param corrmat: Correlation matrix
            :type corrmat: matrix
            :param seed: Random seed to allow replication of samples
            :type seed: int
            :param index_start: Starting value for sample indices
            :type: int
            :returns: matrix -- Parameter samples
          
        """
        if seed:
            self.seed = seed
        # If siz specified, set sample_size
        if siz:
            self.sample_size = siz
        else:
            siz = self.sample_size
        # Take distribution keyword and convert to scipy.stats distribution object
        dists = []
        dist_pars = []
        for p in self.pars.itervalues():
            if p.vary and p.dist != 'discrete':
                eval( 'dists.append(stats.' + p.dist + ')' )
                dist_pars.append(p.dist_pars)
        if len(dists):
            x = lhs(dists, dist_pars, siz=siz, noCorrRestr=noCorrRestr, corrmat=corrmat, seed=seed)
        # Convert 1D array to 1D matrix if only one parameter is varying
        if len(dists) == 1: x = x.reshape((len(x),1))
        for j,p in enumerate(self.pars.values()):
            if p.expr is not None:
                for i,r in enumerate(x):
                    x[i,j] = self.__eval_expr( p.expr, r )
        # Construct sampleset replacing fixed parameters with their 'value' and sampling discrete parameters
        ss = numpy.zeros((siz,len(self.pars)))
        ind = 0
        for i,p in enumerate(self.pars.itervalues()):
            if p.vary and p.dist != 'discrete': 
                ss[:,i] = x[:,ind]
                ind += 1
            elif p.vary and p.dist == 'discrete': 
                dinds = rv_discrete(values=(range(len(p._discrete_vals[0])),p._discrete_vals[1])).rvs(size=siz) 
                for ii,dind in enumerate(dinds):
                    ss[ii,i] = p._discrete_vals[0][dind] 
            else: ss[:,i] = p.value
        return self.create_sampleset( ss, name=name, index_start=index_start )
    def saltelli(self, nsamples, name=None, calc_second_order=True, index_start=1, problem={}):
        """ Create sampleset using Saltelli's extension of the Sobol sequence intended to be used with sobol method. This method calls functionality from the SALib package.
        
            :param nsamples: Number of samples to create for each parameter. If calc_second_order is False, the actual sample size will be N * (D + 2), otherwise, it will be N * (2D + 2)
            :type nsamples: int
            :param name: Name of sample set to be created
            :type name: str
            :param calc_second_order: Calculate second-order sensitivities
            :type calc_second_order: bool
            :param index_start: Starting value for sample indices
            :type index_start: int
            :param problem: Dictionary of model attributes used by sampler
            :type problem: dict
            :param problem: Dictionary of model attributes used by sampler. For example, dictionary with a list with keyname 'groups' containing a list of length of the number of parameters with parameter group names can be used to group parameters with similar effects on the observation.
            :type problem: dict
            :returns: MATK sampleset
          
        """
        try:
            from SALib.sample import saltelli
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import SALib saltelli module. ({})\n".format(exc))
        # Define problem for Saltelli sampler
        problem['num_vars'] = len(self.pars)
        problem['names'] = self.parnames
        problem['bounds'] = zip(self.parmins,self.parmaxs)
        # Create sampleset of saltelli sample
        param_values = saltelli.sample(problem, nsamples, calc_second_order=True)
        # Return sampleset
        return self.create_sampleset( param_values, name=name, index_start=index_start )
    def child( self, in_queue, out_list, reuse_dirs, save, hostname, processor):
        # Ignoring Futurewarning about elementwise comparison for now
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="elementwise")
            for pars,smp_ind,lst_ind in iter(in_queue.get, ('','','')):
                self.workdir_index = smp_ind
                if self.workdir_base is not None:
                    self.workdir = self.workdir_base + '.' + str(self.workdir_index)
                self.parvalues = pars
                status = self.forward(reuse_dirs=reuse_dirs, job_number=smp_ind, hostname=hostname, processor=processor)
                out_list.put([lst_ind, smp_ind, status])
                if not save and not self.workdir is None:
                    rmtree( self.workdir )
                in_queue.task_done()
        in_queue.task_done()
    def parallel(self, parsets, cpus=1, workdir_base=None, save=True,
                reuse_dirs=False, indices=None, verbose=True, logfile=None):

        if not os.name is "posix":
            # Use freeze_support for PCs
            freeze_support()

        # Determine if using working directories or not
        saved_workdir = self.workdir # Save workdir to reset after parallel run
        if not workdir_base is None: self.workdir_base = workdir_base
        if self.workdir_base is None: self.workdir = None

        #if len(hosts) > 0:
        if isinstance( cpus, dict):
            hosts = cpus
            cpus = sum([len(v) for v in hosts.values()])
            processors = [v for l in hosts.values() for v in l]
            hostnames = [k for k,v in hosts.items() for n in v]
            self.cpus = hosts
        elif isinstance(self.cpus,dict) and len(self.cpus) > 0:
            hosts = self.cpus
            cpus = sum([len(v) for v in hosts.values()])
            processors = [v for l in hosts.values() for v in l]
            hostnames = [k for k,v in hosts.items() for n in v]
        elif isinstance(cpus, int):
            hostnames = [None]*cpus
            processors = [None]*cpus
        else:
            print "Error: cpus argument is neither an integer nor a dictionary!"
            return

        # Determine number of samples and adjust cpus if samples < cpus requested
        if isinstance( parsets, numpy.ndarray ): n = parsets.shape[0]
        elif isinstance( parsets, list ): n = len(parsets)
        if n < cpus: cpus = n

        # Start cpus model runs
        resultsq = Queue()
        work = JoinableQueue()
        pool = []
        for i in range(cpus):
            p = Process(target=self.child, args=(work, resultsq, reuse_dirs, save, hostnames[i],processors[i]))
            p.daemon = True
            p.start()
            pool.append(p)

        iter_args = itertools.chain( parsets, ('',)*cpus )
        iter_smpind = itertools.chain( indices, ('',)*cpus )
        iter_lstind = itertools.chain( range(len(parsets)), ('',)*cpus )
        for item in zip(iter_args,iter_smpind,iter_lstind):
            work.put(item)
        
        #if verbose or logfile: 
        #    if logfile: 
        #        f = open(logfile, 'w')
        #        f.write("Number of parameters: %d\n" % len(self.pars) )
        #        f.write("Number of responses: %d\n" % len(self.obs) )
        #    s = "%-8s" % 'index'
        #    for nm in self.parnames:
        #        s += " %22s" % nm
        #    header = True

        results = [[numpy.NAN]]*len(parsets)
        header = True
        for i in range(len(parsets)):
            if logfile and i == 0: 
                f = open(logfile, 'w')
                f.write("Number of parameters: %d\n" % len(self.pars) )
                f.flush()
            lst_ind, smp_ind, resp = resultsq.get()
            if isinstance( resp, str):
                if logfile: 
                    f.write(resp+'\n')
                    f.flush()
            else:
                if isinstance( resp, OrderedDict):
                    self._set_simvalues(resp)
                    results[lst_ind] = resp.values()
                if verbose or logfile:
                    if header:
                        if logfile: 
                            f.write("Number of responses: %d\n" % len(self.obs) )
                        s = "%-8s" % 'index'
                        for nm in self.parnames:
                            s += " %22s" % nm
                        for nm in self.obsnames:
                            s += " %22s" % nm
                        s += '\n'
                        if verbose: print s,
                        if logfile: 
                            f.write( s )
                            f.flush()
                        header = False
                    s = "%-8d" % smp_ind
                    for v in parsets[lst_ind]:
                        s += " %22.16g" % v
                    if results[lst_ind] is not numpy.NAN:
                        for v in results[lst_ind]:
                            if v is None:
                                s += " %22s" % "None"
                            else:
                                s += " %22.16g" % v
                    s += '\n'
                    if verbose: print s,
                    if logfile: 
                        f.write( s )
                        f.flush()
        if logfile: f.close()

        for i in range(len(results)):
            if numpy.any([numpy.isnan(v) for v in results[i]]):
                if len(self.obs) > 0:
                    results[i] = [numpy.NAN]*len(self.obs)

        for p in pool:
            p.join()

        # Clean parent
        self.workdir = saved_workdir
        results = numpy.array(results)
        if results.shape[1] == 1:
            if all(numpy.isnan(r[0]) for r in results):
                results = None

        return results, parsets   
    def parstudy(self, nvals=2, name=None):
        ''' Generate parameter study samples.
            For discrete parameters with nvals>3, bins are chosen to be spaced as far apart as possible, while still being evenly spaced (note that this is based on bins, not actual values). 

        :param name: Name of sample set to be created
        :type name: str
        :param outfile: Name of file where samples will be written. If outfile=None, no file is written.
        :type outfile: str
        :param nvals: number of values for each parameter
        :type nvals: int or list(int)
        :returns: ndarray(fl64) -- Array of samples
        '''

        # Function to help create evenly spaced indices for discrete parameters
        spaced_index = lambda m, n: [i*n//m + n//(2*m) + 1 for i in range(m)]

        if isinstance(nvals,int):
            nvals = [nvals]*len(self.pars)
        x = []
        for p,n in zip(self.pars.values(),nvals):
            if p.dist != 'discrete':
                if n == 1 or not p.vary:
                    if p.value is not None: x.append([p.value])
                    elif p.min is not None and p.max is not None: x.append([(p.max+p.min)/2.])
                    elif p.min is not None: x.append([p.min])
                    elif p.max is not None: x.append([p.max])
                    else: x.append([0.])
                elif n > 1:
                    x.append(numpy.linspace(p.min, p.max, n))
            else:
                if n > len(p.discrete_vals[0]): # If too many values requested, truncate to number of bins
                    print "Warning: Number of values requested for {} is more than the number of its bins ({}). The number of values will be truncated to number of bins.".format(p.name,len(p.discrete_vals))
                    x.append(p.discrete_vals[0])
                elif n == len(p.discrete_vals[0]): x.append(p.discrete_vals[0]) # add all values
                elif n == 1:
                    if p.value: x.append([p.value]) # Just use parameter "value"
                    else: x.append(p.discrete_vals[0][spaced_index(1,len(p.discrete_vals[0]))]) # try to choose middle value
                elif n == 2: # Choose first and last value
                    x.append([p.discrete_vals[0][0], # Choose first
                              p.discrete_vals[0][-1]]) # and last value
                elif n == 3:
                    x.append([p.discrete_vals[0][0], # Choose first
                              p.discrete_vals[0][len(p.discrete_vals[0])/2], # near middle
                              p.discrete_vals[0][len(p.discrete_vals[0])-1]]) # and last value
                else: # Space values as far apart as possible while still being evenly spaced
                    x.append(p.discrete_vals[0][spaced_index(n,len(p.discrete_vals[0]))])

        x = list(itertools.product(*x))
        x = numpy.array(x)

        return self.create_sampleset( x, name=name )
    def fullfact(self,name=None,levels=[]):
        try:
            import pyDOE
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import pyDOE module. ({})".format(exc))
            return
        if len(levels) == 0:
            levels = numpy.array([p.nvals for p in self.pars.values()])
        elif len(levels) != len(self.pars): 
            print "Error: Length of levels ("+str(len(levels))+") not equal to number of parameters ("+str(len(self.pars))+")"
            return
        else:
            levels = numpy.array(levels)
        ds = pyDOE.fullfact(levels)
        mns = numpy.array(self.parmins)
        mxs = numpy.array(self.parmaxs)
        parsets = mns + ds/(levels-1)*(mxs-mns)
        return self.create_sampleset(parsets, name=name)
    def Jac( self, h=None, cpus=1, workdir_base=None,
                    save=True, reuse_dirs=False, verbose=False ):
        ''' Numerical Jacobian calculation

            :param h: Parameter increment, single value or array with npar values
            :type h: fl64 or ndarray(fl64)
            :returns: ndarray(fl64) -- Jacobian matrix
        '''
        try: import lmfit
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import lmfit module. ({})".format(exc))
            return

        # Create lmfit parameter object
        params = lmfit.Parameters()
        for k,p in self.pars.items():
            params.add(k,value=p.value,vary=p.vary,min=p.min,max=p.max,expr=p.expr) 

        return self.__jacobian( params, cpus=cpus, epsfcn=h, workdir_base=workdir_base,verbose=verbose,save=save, reuse_dirs=reuse_dirs)

    def calibrate( self, cpus=1, maxiter=100, lambdax=0.001, minchange=1.0e-16, minlambdax=1.0e-6, verbose=False,
                  workdir=None, reuse_dirs=False, h=1.e-6):
        """ Calibrate MATK model using Levenberg-Marquardt algorithm based on 
            original code written by Ernesto P. Adorio PhD. 
            (UPDEPP at Clarkfield, Pampanga)

            :param cpus: Number of cpus to use
            :type cpus: int
            :param maxiter: Maximum number of iterations
            :type maxiter: int
            :param lambdax: Initial Marquardt lambda
            :type lambdax: fl64
            :param minchange: Minimum change between successive ChiSquares
            :type minchange: fl64
            :param minlambdax: Minimum lambda value
            :type minlambdax: fl4
            :param verbose: If True, additional information written to screen during calibration
            :type verbose: bool
            :returns: best fit parameters found by routine
            :returns: best Sum of squares.
            :returns: covariance matrix
        """
        from minimizer import Minimizer
        fitter = Minimizer(self)
        fitter.calibrate(cpus=cpus,maxiter=maxiter,lambdax=lambdax,minchange=minchange,
                         minlambdax=minlambdax,verbose=verbose,workdir=workdir,reuse_dirs=reuse_dirs,h=h)
    def __eval_expr(self, exprstr, parset):
        aeval = Interpreter()
        for val,nm in zip(parset,self.pars.keys()):
            aeval.symtable[nm] = val
        return aeval(exprstr)
    def MCMC( self, nruns=10000, burn=1000, init_error_std=1., max_error_std=100., verbose=1 ):
        ''' Perform Markov Chain Monte Carlo sampling using pymc package

            :param nruns: Number of MCMC iterations (samples)
            :type nruns: int
            :param burn: Number of initial samples to burn (discard)
            :type burn: int
            :param verbose: verbosity of output
            :type verbose: int
            :param init_error_std: Initial standard deviation of residuals
            :type init_error_std: fl64
            :param max_error_std: Maximum standard deviation of residuals that will be considered
            :type max_error_std: fl64
            :returns: pymc MCMC object
        '''
        if max_error_std < init_error_std:
            print "Error: max_error_std must be greater than or equal to init_error_std"
            return
        try:
            from pymc import Uniform, deterministic, Normal, MCMC, Matplot
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import pymc module. ({})\n".format(exc))
            sys.stderr.write("If pymc is not installed, try installing:\n")
            sys.stderr.write("e.g. try using easy_install: easy_install pymc\n")
        def __mcmc_model( self, init_error_std=1., max_error_std=100. ):
            #priors
            variables = []
            sig = Uniform('error_std', 0.0, max_error_std, value=init_error_std)
            variables.append( sig )
            for nm,mn,mx in zip(self.parnames,self.parmins,self.parmaxs):
                evalstr = "Uniform( '" + str(nm) + "', " +  str(mn) + ", " +  str(mx) + ")"
                variables.append( eval(evalstr) )
            #model
            @deterministic()
            def residuals( pars = variables, p=self ):
                values = []
                for i in range(1,len(pars)):
                    values.append(float(pars[i]))
                pardict = dict(zip(p.parnames,values))
                p.forward(pardict=pardict, reuse_dirs=True)
                return numpy.array(p.residuals())*numpy.array(p.obsweights)
            #likelihood
            y = Normal('y', mu=residuals, tau=1.0/sig**2, observed=True, value=numpy.zeros(len(self.obs)))
            variables.append(y)
            return variables

        M = MCMC( __mcmc_model(self, init_error_std=init_error_std, max_error_std=max_error_std) )
        M.sample(iter=nruns,burn=burn,verbose=verbose)
        return M
    def MCMCplot( self, M ):
        try:
            from pymc import Uniform, deterministic, Normal, MCMC, Matplot
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import pymc module. ({})\n".format(exc))
            sys.stderr.write("If pymc is not installed, try installing:\n")
            sys.stderr.write("e.g. try using easy_install: easy_install pymc\n")
        Matplot.plot(M)
    def emcee( self, lnprob=None, lnprob_args=(), nwalkers=100, nsamples=500, burnin=50, pos0=None, ncpus=1 ):
        ''' Perform Markov Chain Monte Carlo sampling using emcee package

            :param lnprob: Function specifying the natural logarithm of the likelihood function
            :type lnprob: function
            :param nwalkers: Number of random walkers
            :type nwalkers: int
            :param nsamples: Number of samples per walker
            :type nsamples: int
            :param burnin: Number of "burn-in" samples per walker to be discarded
            :type burnin: int
            :param pos0: list of initial positions for the walkers
            :type pos0: list
            :param ncpus: number of cpus
            :type ncpus: int
            :returns: numpy array containing samples
        '''
        try:
            import emcee
        except ImportError as exc:
            sys.stderr.write("Warning: failed to import emcee module. ({})\n".format(exc))
        if lnprob is None:
            lnprob = logposterior(self)
        sampler = emcee.EnsembleSampler(nwalkers, len(self.parnames), lnprob, args=lnprob_args, threads=ncpus)
        if pos0 == None:
            try:
                from pyDOE import lhs
                lh = lhs(len(self.parnames), samples=nwalkers)
                pos0 = []
                for i in range(nwalkers):
                    pos0.append([pmin + (pmax - pmin) * lhval for lhval, pmin, pmax in zip(lh[i], self.parmins, self.parmaxs)])
            except ImportError as exc:
                sys.stderr.write("Warning: failed to import pyDOE module. ({})\n".format(exc))
        pos,prob,state = sampler.run_mcmc(pos0, burnin)
        sampler.reset()
        sampler.run_mcmc(pos, nsamples)
        #return sampler.chain[:, burnin:, :].reshape((-1, len(self.parnames))), sampler.lnprobability[:, burnin:].flatten()
        return sampler
    def differential_evolution(self,bounds=(), workdir=None, strategy='best1bin',maxiter=1000, popsize=15, tol=0.01,
                               mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True,
                               init='latinhypercube',save_evals=False):
        ''' Perform differential evolution calibration using scipy.optimize.differential_evolution:

            Differential Evolution is stochastic in nature (does not use gradient
            methods) to find the minimium, and can search large areas of candidate
            space, but often requires larger numbers of function evaluations than
            conventional gradient based techniques.

            The algorithm is due to Storn and Price.

            Parameters
            func : callable
            The objective function to be minimized.  Must be in the form
            ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
            and ``args`` is a  tuple of any additional fixed parameters needed to
            completely specify the function.
            bounds : sequence
            Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
            defining the lower and upper bounds for the optimizing argument of
            `func`. It is required to have ``len(bounds) == len(x)``.
            ``len(bounds)`` is used to determine the number of parameters in ``x``.
            strategy : str, optional
            The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
            The default is 'best1bin'.
            maxiter : int, optional
            The maximum number of generations over which the entire population is
            evolved. The maximum number of function evaluations (with no polishing)
            is: ``(maxiter + 1) * popsize * len(x)``
            popsize : int, optional
            A multiplier for setting the total population size.  The population has
            ``popsize * len(x)`` individuals.
            tol : float, optional
            When the mean of the population energies, multiplied by tol,
            divided by the standard deviation of the population energies
            is greater than 1 the solving process terminates:
            ``convergence = mean(pop) * tol / stdev(pop) > 1``
            mutation : float or tuple(float, float), optional
            The mutation constant. In the literature this is also known as
            differential weight, being denoted by F.
            If specified as a float it should be in the range [0, 2].
            If specified as a tuple ``(min, max)`` dithering is employed. Dithering
            randomly changes the mutation constant on a generation by generation
            basis. The mutation constant for that generation is taken from
            ``U[min, max)``. Dithering can help speed convergence significantly.
            Increasing the mutation constant increases the search radius, but will
            slow down convergence.
            recombination : float, optional
            The recombination constant, should be in the range [0, 1]. In the
            literature this is also known as the crossover probability, being
            denoted by CR. Increasing this value allows a larger number of mutants
            to progress into the next generation, but at the risk of population
            stability.
            seed : int or `np.random.RandomState`, optional
            If `seed` is not specified the `np.RandomState` singleton is used.
            If `seed` is an int, a new `np.random.RandomState` instance is used,
            seeded with seed.
            If `seed` is already a `np.random.RandomState instance`, then that
            `np.random.RandomState` instance is used.
            Specify `seed` for repeatable minimizations.
            disp : bool, optional
            Display status messages
            callback : callable, `callback(xk, convergence=val)`, optional
            A function to follow the progress of the minimization. ``xk`` is
            the current value of ``x0``. ``val`` represents the fractional
            value of the population convergence.  When ``val`` is greater than one
            the function halts. If callback returns `True`, then the minimization
            is halted (any polishing is still carried out).
            polish : bool, optional
            If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
            method is used to polish the best population member at the end, which
            can improve the minimization slightly.
            init : string, optional
            Specify how the population initialization is performed. Should be
            one of:

                - 'latinhypercube'
                - 'random'

            The default is 'latinhypercube'. Latin Hypercube sampling tries to
            maximize coverage of the available parameter space. 'random' initializes
            the population randomly - this has the drawback that clustering can
            occur, preventing the whole of parameter space being covered.
            Returns
            -------
            res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing, then
            OptimizeResult also contains the ``jac`` attribute.
        '''
        try: from scipy.optimize import differential_evolution
        except ImportError as exc:
            sys.stderr.write("Error: failed to import scipy.optimize.differential_evolution module. ({})".format(exc))
            return
        if save_evals: self._minimize_pars = []; self._minimize_sims = []
        if len(bounds) == 0:
            bounds = zip(self.parmins,self.parmaxs)
        res = differential_evolution(self.__ssr,bounds,args=(workdir,save_evals),strategy=strategy,maxiter=maxiter,popsize=popsize,
                                     tol=tol,mutation=mutation,recombination=recombination,seed=seed,callback=callback,disp=disp,
                                     polish=polish,init=init)
        if save_evals:
            return res, self.create_sampleset(self._minimize_pars, responses=self._minimize_sims)
        else: return res
    def __ssr(self, x, workdir, save_evals):
        pardict = dict([(k,n) for k,n in zip(self.parnames,x)])
        self.forward(pardict=pardict,workdir=workdir,reuse_dirs=True)
        if save_evals:
            self._minimize_pars.append(self.parvalues)
            self._minimize_sims.append(self.simvalues)
        return self.ssr()

class logposterior(object):
    def __init__(self, prob, var=1):
        self.prob = prob
        self.mins = prob.parmins
        self.maxs = prob.parmaxs
        self.var = var
    def logprior(self,ts):
        for mn,mx,t in zip(self.mins,self.maxs,ts):
            if mn > t or t > mx: return -numpy.inf 
        return 0.0
    def loglhood(self,ts):
        pardict = dict(zip(self.prob.parnames, ts))
        self.prob.forward(pardict=pardict, reuse_dirs=True)
        return -0.5*(numpy.sum((numpy.array(self.prob.residuals()))**2)) / self.var - numpy.log(self.var)
    def __call__(self, ts):
        lpri = self.logprior(ts)
        if lpri == -numpy.inf:
            return lpri
        else:
            return lpri + self.loglhood(ts)

class logposteriorwithunknownvariance(logposterior):
    def __init__(self, prob, var="var"):
        self.prob = prob
        self.mins = prob.parmins
        self.maxs = prob.parmaxs
        self.var = var
    def loglhood(self,ts):
        pardict = dict(zip(self.prob.parnames, ts))
        self.prob.forward(pardict=pardict, reuse_dirs=True)
        #print "ts: " + str(ts)
        #print "ssr: " + str(numpy.sum((numpy.array(self.prob.residuals()))**2))
        #print zip(self.prob.simvalues, self.prob.obsvalues)
        #return -0.5*(numpy.sum((numpy.array(self.prob.residuals()))**2)) / self.prob.pars[self.var].value - numpy.log(self.prob.pars[self.var].value)
        return -0.5*(numpy.sum((numpy.array(self.prob.residuals()))**2)) / self.prob.pars[self.var].value - (len(self.prob.obs)/2)*numpy.log(self.prob.pars[self.var].value)

class logposteriorwithvariance(object):
    def __init__(self, prob, var=1):
        self.prob = prob
        self.mins = prob.parmins
        self.maxs = prob.parmaxs
        self.var = var
        #print prob
    def logprior(self,ts):
        for mn,mx,t in zip(self.mins,self.maxs,ts):
            if mn > t or t > mx: return -numpy.inf
        return 0.0
    def loglhood(self,ts):
        pardict = dict(zip(self.prob.parnames, ts))
        self.prob.forward(pardict=pardict, reuse_dirs=True)
        #return -0.5*(numpy.sum((numpy.array(self.prob.residuals()))**2)) / self.var - numpy.log(self.var)
        #print self.prob.residuals()
        #print self.var
        #print numpy.array(self.prob.residuals() / self.var)
        #print -0.5*(numpy.sum((numpy.array(self.prob.residuals() / self.var))**2))
        return -0.5*(numpy.sum((numpy.array(self.prob.residuals() / self.var))**2))
    def __call__(self, ts):
        lpri = self.logprior(ts)
        if lpri == -numpy.inf:
            return lpri
        else:
            #print "Hello4"
            return lpri + self.loglhood(ts)
