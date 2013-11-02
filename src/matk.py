import sys, os
from parameter import Parameter
from observation import Observation
from sampleset import SampleSet
#import pesting
#import dakoting
#import calibrate
#from sample import *
import numpy 
from lhs import *
import cPickle as pickle
from shutil import rmtree
import itertools
from multiprocessing import Process, Manager, freeze_support

class matk(object):
    """ Class for Model Analysis ToolKit (MATK) module
    """
    def __init__(self, **kwargs):
        '''Initialize MATK object
        :returns: object -- MATK object
        '''
        self.model = ''
        self.model_args = None
        self.model_kwargs = None
        self.ncpus = 1
        self.workdir_base = None
        self.workdir = None
        self.templatedir = None
        self.results_file = None
        self.parameters_file = None
        self.seed = None
        self.sample_size = 10 
        for k,v in kwargs.iteritems():
            if 'sample_size' == k:
                self.sample_size = int(v)
            elif 'ncpus' == k:
                self.ncpus == int(v)
            elif 'workdir_base' == k:
                self.workdir_base = v
            elif 'workdir' == k:
                self.workdir = v
            elif 'templatedir' == k:
                self.templatedir = v
            elif 'model' == k:
                self.model = v
            elif 'model_args' == k:
                self.model_args = v
            elif 'model_kwargs' == k:
                self.model_kwargs = v
            elif 'parameters_file' == k:
                self.parameters_file = v
            elif 'results_file' == k:
                self.results_file = v
            elif 'templatedir' == k:
                self.templatedir = v
            else:
                print k + ' is not a valid argument'
      
        self._parlist = []
        self._obslist = []
        self._samplesetlist = []
        self.workdir_index = 0
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
    def templatedir(self):
        """ Set the name of the templatedir for parallel runs   
        """
        return self._templatedir
    @templatedir.setter
    def templatedir(self,value):
        self._templatedir = value
    @property
    def parameters_file(self):
        """ Set the name of the parameters_file for parallel runs   
        """
        return self._parameters_file
    @parameters_file.setter
    def parameters_file(self,value):
        self._parameters_file = value
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
    @property
    def parlist(self):
        return self._parlist
    @property
    def par(self):
        return dict([[p.name,p] for p in self.parlist if p.name])
    @property
    def obslist(self):
        return self._obslist
    @property
    def obs(self):
        return dict([[o.name,o] for o in self.obslist if o.name])
    @property
    def samplesetlist(self):
        return self._samplesetlist
    @property
    def sampleset(self):
        return dict([[s.name,s] for s in self.samplesetlist if s.name])
    def add_par(self, name, **kwargs):
        """ Add parameter to problem

            :param name: Name of parameter
            :type name: str
            :param kwargs: keyword arguments passed to parameter class
        """
        if name in self.par: 
            for i in range(len(self.parlist)):
                if self.parlist[i].name == name:
                    del self.parlist[i]
                    break
        self.parlist.append(Parameter(name,**kwargs))
    def add_obs(self,name,**kwargs):
        """ Add observation to problem
            
            :param name: Name of observation
            :type name: str
            :param kwargs: keyword arguments passed to observation class
        """
        if name in self.obs: 
            for i in range(len(self.obslist)):
                if self.obslist[i].name == name:
                    del self.obslist[i]
                    break
        self.obslist.append(Observation(name,**kwargs))
    def add_sampleset(self,name,samples,responses=None,indices=None,index_start=1):
        """ Add sample set to problem
            
            :param name: Name of sample set
            :type name: str
            :param samples: Matrix of parameter samples with npar columns in order of [p.name for p in matkobj.parlist] 
            :type samples: list(fl64),ndarray(fl64)
            :param responses: Matrix of associated responses with nobs columns in order of [o.name for o in matkobj.obslist] if observation exists (existence of observations is not required) 
            :type responses: list(fl64),ndarray(fl64)
            :param indices: Sample indices to use when creating working directories and output files
            :type indices: list(int),ndarray(int)
        """
        if not isinstance( samples, (list,numpy.ndarray)):
            print "Error: Parameter samples are not a list or ndarray"
            return 1
        npar = len(self.parlist)
        # If list, convert to ndarray
        if isinstance( samples, list ):
            samples = numpy.array(samples)
        if not samples.shape[1] == npar:
            print "Error: The number of columns in sample is not equal to the number of parameters in the problem"
            return 1
        # Delete old sampleset with same name if it exists
        if name in self.sampleset: 
            for i in range(len(self.samplesetlist)):
                if self.samplesetlist[i].name == name:
                    del self.samplesetlist[i]
                    break
        if len(self.parlist) > 0:
            parnames = self.get_par_names()
        else:
            parnames = None
        if len(self.obslist) > 0:
            obsnames = self.get_obs_names()
        else:
            obsnames = None
        self.samplesetlist.append(SampleSet(name,samples,responses=responses,indices=indices,index_start=index_start,
                                           parnames=parnames, obsnames=obsnames))
    def get_sims(self):
        """ Get the current simulated values
            :returns: lst(fl64) -- simulated values in order of matk.obslist
        """
        return [obs.sim for obs in self.obslist]
    def set_obs_values(self, *args, **kwargs):
        """ Set simulated values using a dictionary or keyword arguments
        """
        if len(args) > 0 and len(kwargs) > 0:
            print "Warning: dictionary arg will overide keyword args"
        if len(args) > 0:
            if isinstance( args[0], dict ):
                obsdict = self.obs
                for k,v in args[0].iteritems():
                    if k in obsdict:
                        obsdict[k].value = v
                    else:
                        self.add_obs( k, value=v ) 
            elif isinstance( args[0], (list,tuple,numpy.ndarray) ):
                if isinstance( args[0], (list,tuple) ):
                    if not len(args[0]) == len(self.obslist): 
                        print "Error: Number of simulated values in list or tuple does not match created observations"
                        return
                elif isinstance( args[0], numpy.ndarray ):
                    if not args[0].shape[0] == len(self.obslist): 
                        print "Error: Number of simulated values in ndarray does not match created observations"
                        return
                i = 0
                for v in args[0]:
                    self.obslist[i].value = v
                    i += 1
        else:
            obsdict = self.obs
            for k,v in kwargs.iteritems():
                if k in obsdict:
                    obsdict[k].value = v
                else:
                    self.add_obs( k, value=v ) 
    def _set_sims(self, *args, **kwargs):
        """ Set simulated values using a dictionary or keyword arguments
        """
        if len(args) > 0 and len(kwargs) > 0:
            print "Warning: dictionary arg will overide keyword args"
        if len(args) > 0:
            if isinstance( args[0], dict ):
                obsdict = self.obs
                for k,v in args[0].iteritems():
                    if k in obsdict:
                        obsdict[k].sim = v
                    else:
                        self.add_obs( k, sim=v ) 
            elif isinstance( args[0], (list,tuple,numpy.ndarray) ):
                if isinstance( args[0], (list,tuple) ):
                    if not len(args[0]) == len(self.obslist): 
                        print "Error: Number of simulated values in list or tuple does not match created observations"
                        return
                elif isinstance( args[0], numpy.ndarray ):
                    if not args[0].shape[0] == len(self.obslist): 
                        print "Error: Number of simulated values in ndarray does not match created observations"
                        return
                i = 0
                for v in args[0]:
                    self.obslist[i].sim = v
                    i += 1
        else:
            obsdict = self.obs
            for k,v in kwargs.iteritems():
                if k in obsdict:
                    obsdict[k].sim = v
                else:
                    self.add_obs( k, sim=v ) 
    def set_par_values(self,*args, **kwargs):
        """ Set parameters using values in first argument
        """
        if len(args[0]) > 0 and len(kwargs) > 0:
            print "Warning: dictionary arg will overide keyword args"
        if len(args[0]) > 0:
            if isinstance( args[0], dict ):
                pardict = self.par
                for k,v in args[0].iteritems():
                    pardict[k].value = v
            elif isinstance( args[0], (list,tuple,numpy.ndarray)):
                if isinstance( args[0], (list,tuple)):
                    if not len(args[0]) == len(self.parlist): 
                        print "Error: Number of parameter values in list or tuple does not match created parameters"
                        return
                elif isinstance( args[0], numpy.ndarray ):
                    if not args[0].shape[0] == len(self.parlist): 
                        print "Error: Number of parameter values in ndarray does not match created parameters"
                        return
                i = 0
                for v in args[0]:
                    self.parlist[i].value = v
                    i += 1
        else:
            for k,v in kwargs.iteritems():
                self.par[k].value = v
    def get_par_values(self):
        """ Get parameter values
        """
        return [par.value for par in self.parlist]
    def get_par_names(self):
        """ Get parameter names
        """
        return [par.name for par in self.parlist]
    def get_par_nvals(self):
        """ Get parameter nvals (number of values for parameter studies)
        """
        return [par.nval for par in self.parlist]
    def get_obs_values(self):
        """ Get observation values
        """
        return [o.value for o in self.obslist]
    def get_obs_names(self):
        """ Get observation names
        """
        return [o.name for o in self.obslist]
    def get_residuals(self):
        """ Get least squares values
        """
        return [o.residual for o in self.obslist]
    def get_par_mins(self):
        """ Get parameter lower bounds
        """
        return [par.min for par in self.parlist]
    def get_par_maxs(self):
        """ Get parameter lower bounds
        """
        return [par.max for par in self.parlist]
    def get_par_dists(self):
        """ Get parameter probabilistic distributions
        """
        return [par.dist for par in self.parlist]
    def get_par_dist_pars(self):
        """ Get parameters needed by parameter distributions
        """
        return [par.dist_pars for par in self.parlist]
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
    def forward(self, pardict=None, workdir=None, reuse_dirs=False):
        """ Run MATK model using current values

            :param pardict: Dictionary of parameter values keyed by parameter names
            :type pardict: dict
            :param workdir: Name of directory where model will be run. It will be created if it does not exist
            :type workdir: str
            :param reuse_dirs: If True and workdir exists, the model will reuse the directory
            :type reuse_dirs: bool
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
        if hasattr( self.model, '__call__' ):
            if pardict is None:
                pardict = dict([(par.name,par.value) for par in self.parlist])
            if self.model_args is None and self.model_kwargs is None:
                sims = self.model( pardict )
            elif not self.model_args is None and self.model_kwargs is None:
                sims = self.model( pardict, *self.model_args )
            elif self.model_args is None and not self.model_kwargs is None:
                sims = self.model( pardict, **self.model_kwargs )
            elif not self.model_args is None and not self.model_kwargs is None:
                sims = self.model( pardict, *self.model_args, **self.model_kwargs )
            self._set_sims(sims)
        else:
            print "Error: Model is not a Python function"
            return 1
        if not curdir is None:
            os.chdir( curdir )
        return 0
    def calibrate(self):
        """ Calibrate MATK model
        """
        x,cov_x,infodic,mesg,ier = calibrate.least_squares(self)
        return x,cov_x,infodic,mesg,ier
    def set_lhs_samples(self, name, siz=None, noCorrRestr=False, corrmat=None, seed=None, index_start=1):
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
        for dist in self.get_par_dists():
            eval( 'dists.append(stats.' + dist + ')' )
        dist_pars = self.get_par_dist_pars()
        x = lhs(dists, dist_pars, siz=siz, noCorrRestr=noCorrRestr, corrmat=corrmat, seed=seed)
        self.add_sampleset( name, x, index_start=index_start )
    def run_samples(self, name=None, ncpus=1, templatedir=None, workdir_base=None,
                    save=True, reuse_dirs=False ):
        """ Run model using values in samples for parameter values
            If samples are not specified, LHS samples are produced
            
            :param name: Name of MATK sample set object
            :type samples: str
            :param ncpus: number of cpus to use to run models concurrently
            :type ncpus: int
            :param templatedir: Name of folder including files needed to run model (e.g. template files, instruction files, executables, etc.)
            :type templatedir: str
            :param workdir_base: Base name for model run folders, run index is appended to workdir_base
            :type workdir_base: str
            :param save: If True, model files and folders will not be deleted during parallel model execution
            :type save: bool
            :param reuse_dirs: Will use existing directories if True, will return an error if False and directory exists
            :type reuse_dirs: bool
            :returns: tuple(ndarray(fl64),ndarray(fl64)) - (Matrix of responses from sampled model runs siz rows by npar columns, Parameter samples, same as input samples if provided)
            
        """
        if name == None and len(self.sampleset) == 1:
            name = self.sampleset[0].name
        if templatedir:
            self.templatedir = templatedir
        if workdir_base:
            self.workdir_base = workdir_base
                
        if ncpus == 1:
            out = []
            for sample, index in zip(self.sampleset[name].samples,self.sampleset[name].indices):
                self.set_par_values(sample)
                if not self.workdir_base is None:
                    workdir = self.workdir_base + '.' + str(index)
                else:
                    workdir = None
                self.forward(workdir=workdir,reuse_dirs=reuse_dirs)
                if not save:
                    rmtree( workdir )
                responses = self.get_sims()
                out.append( responses )
            out = numpy.array(out)
        elif ncpus > 1:
            out, samples = self.parallel_mp(ncpus, self.sampleset[name].samples, indices=self.sampleset[name].indices,
                                         templatedir=templatedir, workdir_base=workdir_base, 
                                         save=save, reuse_dirs=reuse_dirs)
        else:
            print 'Error: number of cpus (ncpus) must be greater than zero'
            return
        self.sampleset[name].responses = out 
    def parallel(self, ncpus, par_sets, templatedir=None, workdir_base=None, save=True,
                reuse_dirs=False, indices=None):
 
        def child( prob ):
            if hasattr( prob.model, '__call__' ):
                status = prob.forward(reuse_dirs=reuse_dirs)
                if status:
                    print "Error running forward model for parallel job " + str(prob.workdir_index)
                    os._exit( 0 )
                out = dict( zip(prob.get_obs_names(),prob.get_sims()) )
                if self.workdir is None:
                    pickle.dump( out, open(self.results_file, "wb"))
                else:
                    pickle.dump( out, open(os.path.join(self.workdir,self.results_file), "wb"))
            os._exit( 0 )

        def set_child( prob ):
            if prob.workdir_base is not None:
                prob.workdir = prob.workdir_base + '.' + str(prob.workdir_index)
            prob.results_file = 'output' + '.' + str(prob.workdir_index)

        # Determine if using working directories or not
        saved_workdir = self.workdir # Save workdir to reset after parallel run
        if not workdir_base is None: self.workdir_base = workdir_base
        if self.workdir_base is None: self.workdir = None

        # Determine number of samples and adjust ncpus if samples < ncpus requested
        if isinstance( par_sets, numpy.ndarray ): n = par_sets.shape[0]
        elif isinstance( par_sets, list ): n = len(par_sets)
        if n < ncpus: ncpus = n

        # Start ncpus model runs
        jobs = []
        pids = []
        workdirs = []
        ps_index = [] # Parset index used for reordering
        results_files = []
        for i in range(ncpus):
            self.workdir_index = indices[i]
            set_child( self )
            pardict = dict(zip(self.get_par_names(), par_sets[i] ) )
            self.set_par_values(pardict)
            pid = os.fork()
            if pid:
                parent = True
                pids.append(pid)
                workdirs.append( self.workdir )
                ps_index.append( i )
                results_files.append(self.results_file)
            else:
                parent = False
                child( self )
                os._exit( 0 )

        
        # Wait for jobs and start new ones
        if parent:
            # Create dictionaries of results_file names and working directories for reference below
            resfl_dict = dict(zip(pids,results_files)) 
            wkdir_dict = dict(zip(pids,workdirs))
            ps_index_dict = dict(zip(pids,ps_index))
            res_index = []
            responses = []
            njobs_started = ncpus
            njobs_finished = 0
            while njobs_finished < n and parent:
                    rpid,status = os.wait() # Wait for jobs
                    if status:
                        print os.strerror( status )
                        return 1
                    # Update dictionaries to include any new jobs
                    resfl_dict = dict(zip(pids,results_files))
                    wkdir_dict = dict(zip(pids,workdirs))
                    ps_index_dict = dict(zip(pids,ps_index))
                    # Load results from completed job
                    if not wkdir_dict[rpid] is None:
                        out = pickle.load( open( os.path.join(wkdir_dict[rpid],resfl_dict[rpid]), "rb" ) )
                        if save is False:
                            rmtree( wkdir_dict[rpid] )
                    else:
                        out = pickle.load( open( resfl_dict[rpid], "rb" ))
                        if save is False:
                            os.remove( resfl_dict[rpid] )
                    self._set_sims( out )
                    responses.append( self.get_sims() )
                    res_index.append( ps_index_dict[rpid] )
                    njobs_finished += 1
                    # Start new jobs
                    if njobs_started < n:
                        njobs_started += 1
                        self.workdir_index = indices[njobs_started-1]
                        set_child( self )
                        pardict = dict(zip(self.get_par_names(), par_sets[njobs_started-1] ) )
                        self.set_par_values(pardict)
                        pid = os.fork()
                        if pid:
                            parent = True
                            pids.append(pid)
                            workdirs.append( self.workdir)
                            results_files.append(self.results_file)
                            ps_index.append( njobs_started-1 )
                        else:
                            parent = False
                            child( self )
                            os._exit( 0 )
        
        # Rearrange responses to correspond with par_sets
        res = []
        for i in range(n):
            res.append( responses[res_index[i]] ) 
        responses = numpy.array(res)

        # Clean parent
        self.workdir = saved_workdir

        return responses, par_sets   
    def parallel_mp(self, ncpus, par_sets, templatedir=None, workdir_base=None, save=True,
                reuse_dirs=False, indices=None):

        if not os.name is "posix":
            # Use freeze_support for PCs
            freeze_support()
 
        def child( prob, in_queue, out_list, reuse_dirs, save):
            pars,smp_ind,lst_ind = in_queue.get()
            if pars == None:
                return
            prob.workdir_index = ind
            set_child( prob )
            prob.set_par_values( pars )
            status = prob.forward(reuse_dirs=reuse_dirs)
            if status:
                print "Error running forward model for parallel job " + str(prob.workdir_index)
            else:
                out_list[lst_ind] = prob.get_sims()
            if not save and not prob.workdir is None:
                rmtree( prob.workdir )

        def set_child( prob ):
            if prob.workdir_base is not None:
                prob.workdir = prob.workdir_base + '.' + str(prob.workdir_index)

        # Determine if using working directories or not
        saved_workdir = self.workdir # Save workdir to reset after parallel run
        if not workdir_base is None: self.workdir_base = workdir_base
        if self.workdir_base is None: self.workdir = None

        # Determine number of samples and adjust ncpus if samples < ncpus requested
        if isinstance( par_sets, numpy.ndarray ): n = par_sets.shape[0]
        elif isinstance( par_sets, list ): n = len(par_sets)
        if n < ncpus: ncpus = n

        # Start ncpus model runs
        manager = Manager()
        results = manager.list(range(len(par_sets)))
        work = manager.Queue(ncpus)
        pool = []
        for pars,ind in zip(par_sets,indices):
            p = Process(target=child, args=(self, work, results, reuse_dirs, save))
            p.start()
            pool.append(p)
            
        iter_args = itertools.chain( par_sets, (None,)*ncpus )
        iter_smpind = itertools.chain( indices, (None,)*ncpus )
        iter_lstind = itertools.chain( range(len(par_sets)), (None,)*ncpus )
        for item in zip(iter_args,iter_smpind,iter_lstind):
            work.put(item)
        
        for p in pool:
            p.join()

        # Clean parent
        self.workdir = saved_workdir
        results = numpy.array(results)

        return results, par_sets   
    def set_parstudy_samples(self, name, *args, **kwargs):
        ''' Generate parameter study samples
        
        :param name: Name of sample set to be created
        :type name: str
        :param outfile: Name of file where samples will be written. If outfile=None, no file is written.
        :type outfile: str
        :param *args: Number of values for each parameter. The order is expected to match order of matk.parlist (e.g. [p.name for p in matk.parlist])
        :type *args: tuple(fl64), list(fl64), or ndarray(fl64)
        :param **kwargs: keyword arguments where keyword is the parameter name and argument is the number of desired values
        :type **kwargs: dict(fl64)
        :returns: ndarray(fl64) -- Array of samples
        '''
        outfile = None
        for k,v in kwargs.iteritems():
            if k == 'outfile':
                outfile = v

        if len(args) > 0 and len(kwargs) > 0:
            print "Warning: dictionary arg will overide keyword args"
        if len(args) > 0:
            if isinstance( args[0], dict ):
                pardict = self.par
                for k,v in args[0].iteritems():
                    pardict[k].nvals = v
            elif isinstance( args[0], (list,tuple,numpy.ndarray)):
                if isinstance( args[0], (list,tuple)):
                    if not len(args[0]) == len(self.parlist): 
                        print "Error: Number of values in list or tuple does not match created parameters"
                        return
                elif isinstance( args[0], numpy.ndarray ):
                    if not args[0].shape[0] == len(self.parlist): 
                        print "Error: Number of values in ndarray does not match created parameters"
                        return
                i = 0
                for v in args[0]:
                    self.parlist[i].nvals = v
                    i += 1
        else:
            for k,v in kwargs.iteritems():
                if not k == 'outfile':
                    self.par[k].nvals = v


        x = []
        for p in self.parlist:
            if p.nvals == 1:
                x.append(numpy.linspace(p.value, p.max, p.nvals))
            if p.nvals > 1:
                x.append(numpy.linspace(p.min, p.max, p.nvals))

        x = list(itertools.product(*x))
        x = numpy.array(x)

        self.add_sampleset( name, x )

    def save_sampleset( self, outfile, sampleset ):
        ''' Save sampleset to file

            :param outfile: Name of file where sampleset will be written
            :type outfile: str
            :param sampleset: Sampleset name
            :type sampleset: str
        '''

        if isinstance( sampleset, str ):
            x = numpy.column_stack([self.sampleset[sampleset].indices,self.sampleset[sampleset].samples])
            if not self.sampleset[sampleset].responses is None:
                x = numpy.column_stack([x,self.sampleset[sampleset].responses])
        else:
            print 'Error: sampleset is not a string'
            return

        if outfile:
            f = open(outfile, 'w')
            f.write("%-8s" % 'index' )
            # Print par names
            for nm in self.get_par_names():
                f.write(" ")
                f.write("%15s" % nm )
            # Print obs names if responses exist
            if not self.sampleset[sampleset].responses is None:
                for nm in self.get_obs_names():
                    f.write(" ")
                    f.write("%15s" % nm )
            f.write('\n')
            for row in x:
                f.write("%-8d" % row[0] )
                for i in range(1,len(row)):
                    f.write("%16lf" % row[i] )
                f.write('\n')


            #numpy.savetxt(f, x, fmt='%16lf')
            f.close()


