import sys, os
from parameter import Parameter
from observation import Observation
#import pesting
#import dakoting
#import calibrate
#from sample import *
import numpy 
from lhs import *
import cPickle as pickle
from shutil import rmtree
import itertools

class matk(object):
    """ Class for Model Analysis ToolKit module
    """
    def __init__(self, **kwargs):
        self.model = ''
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
        self.workdir_index = 0
    @property
    def model(self):
        """ Python function or system command to run model
        """
        return self._model
    @model.setter
    def model(self,value):
        self._model = value       
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
        """ Set the name of the templatedir for parallel runs   
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
    def add_par(self, name, **kwargs):
        """ Add parameter to problem
        """
        self.parlist.append(Parameter(name,**kwargs))
    def add_obs(self,name,**kwargs):
        """ Add observation to problem
        """
        self.obslist.append(Observation(name,**kwargs))
    def get_sims(self):
        """ Get the current simulated values
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
    def write_model_files(self, workdir=None):
        """ Write model files with current parameters"""
        if self.flag['pest']:
            pesting.write_pest_files(self, workdir)
        if self.flag['dakota']:
            dakoting.write_dakota_files(self, workdir)
    def read_model_files(self, workdir=None):
        """ Write model files with current parameters"""
        if self.flag['pest']:
            pesting.read_pest_files(self,workdir)
        if self.flag['dakota']:
            dakoting.read_dakota_files(self,workdir)
    def _run_model(self):
        """ Run simulation command on system"""
        if hasattr( self.model, '__call__' ):
            pardict = self.par
            sims = self.model( pars.dict )
            self.set_sims(sims)
        else:
            run_model(self.sim_command)
    def forward(self, workdir=None):
        """ Run pymads problem forward model using current values
        """
        if not workdir is None: self.workdir = workdir
        if not self.workdir is None:
            curdir = os.getcwd()
            if not os.path.isdir( self.workdir ):
                os.makedirs( self.workdir )
            else:
                print "Error: " + self.workdir + " already exists"
                return 1
            os.chdir( self.workdir )
        if hasattr( self.model, '__call__' ):
            pardict = dict([(par.name,par.value) for par in self.parlist])
            sims = self.model( pardict )
            self._set_sims(sims)
        else:
            run_model(self.sim_command)
        if not self.workdir is None:
            os.chdir( curdir )
        return 0
    def run_parallel(self):
        """ Run models concurrently on multiprocessor machine
        """
        if not self.flag['parallel']:
            print 'Parallel execution not enabled, set ncpus to number of processors'
            return 0
        #run_model.parallel(self)
    def calibrate(self):
        """ Calibrate pymads problem model
        """
        x,cov_x,infodic,mesg,ier = calibrate.least_squares(self)
        return x,cov_x,infodic,mesg,ier
    def get_samples(self, siz=None, noCorrRestr=False, corrmat=None, outfile=None, seed=None):
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
                If outfile=None, no file is written.
            seed : int
                random seed to allow replication of samples
            
            Returns
            -------
            samples : ndarray 
                Parameter samples
          
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
        x =  numpy.array(x).transpose()
        if outfile:
            f = open(outfile, 'w')
            for nm in self.get_par_names():
                f.write(" ")
                f.write("%16s" % nm )
            f.write('\n')
            numpy.savetxt(f, x, fmt='%16lf')
        return x
    def run_samples(self, siz=None, noCorrRestr=False, corrmat=None,
                    samples=None, outfile=None, parallel=False, ncpus=1,
                    templatedir=None, workdir_base=None, seed=None,
                    save=True, index_start=1 ):
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
            save : bool
                if True, model files and folders will not be deleted
                during parallel model execution
            index_start : int
                The initial index to be appended to working directories
                and output files

            Returns
            -------
            responses : ndarray 
                Responses from model runs
            samples : ndarray 
                Parameter samples, same as input samples if provided
            
        """
        if seed:
            self.seed = seed
        if siz:
            self.sample_size = siz
        if samples == None:
            samples = self.get_samples(siz, noCorrRestr=noCorrRestr, corrmat=corrmat, seed=seed)
        if parallel:
            if not ncpus:
                print 'Number of cpus is not set for parallel model runs, use option ncpus'
                return 1
            if templatedir:
                self.templatedir = templatedir
            if workdir_base:
                self.workdir_base = workdir_base
                
        if not parallel:
            out = []
            for sample in samples:
                self.set_par_values(sample)
                self.forward()
                responses = self.get_sims()
                out.append( responses )
        else:
            out, samples = self.parallel(ncpus, samples, templatedir=templatedir, workdir_base=workdir_base,
                                        save=save, index_start=index_start)
        if outfile:
            f = open(outfile, 'w')
            f.write( '%-9s '%'id ' )
            for parnm in self.get_par_names():
                f.write( '%22s '%parnm)
            for obsnm in self.get_obs_names():
                f.write( '%22s '%obsnm)
            f.write( '\n')
            for sid in range(self.sample_size):
                f.write( '%-9d '%(int(sid) + 1))
                for val in samples[sid]:
                    f.write( '%22.16e '% val)
                for val in responses[sid]:
                    f.write( '%22.16e '% val)
                f.write( '\n')
            f.close()
        return out, samples
    def parallel(self, ncpus, par_sets, templatedir=None, workdir_base=None, save=True, index_start=1 ):
 
        def child( prob ):
            if hasattr( prob.model, '__call__' ):
                status = prob.forward()
                if status:
                    print "Error running forward model for parallel job " + str(prob.workdir_index)
                    os._exit( 1 )
                out = prob.get_sims()
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
        results_files = []
        for i in range(ncpus):
            self.workdir_index = index_start + i
            set_child( self )
            pardict = dict(zip(self.get_par_names(), par_sets[i] ) )
            self.set_par_values(pardict)
            pid = os.fork()
            if pid:
                parent = True
                pids.append(pid)
                workdirs.append( self.workdir )
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
            njobs_started = ncpus
            njobs_finished = 0
            responses = []
            while njobs_finished < n and parent:
                    rpid,status = os.wait() # Wait for jobs
                    if status:
                        #print os.strerror( status )
                        return 1
                    # Update dictionaries to include any new jobs
                    resfl_dict = dict(zip(pids,results_files))
                    wkdir_dict = dict(zip(pids,workdirs))
                    # Load results from completed job
                    if not wkdir_dict[rpid] is None:
                        out = pickle.load( open( os.path.join(wkdir_dict[rpid],resfl_dict[rpid]), "rb" ) )
                        if save is False:
                            rmtree( wkdir_dict[rpid] )
                    else:
                        out = pickle.load( open( resfl_dict[rpid], "rb" ))
                        if save is False:
                            os.remove( resfl_dict[rpid] )
                    responses.append(out)
                    njobs_finished += 1
                    # Start new jobs
                    if njobs_started < n:
                        njobs_started += 1
                        self.workdir_index = index_start + njobs_started - 1
                        set_child( self )
                        pardict = dict(zip(self.get_par_names(), par_sets[njobs_started-1] ) )
                        self.set_par_values(pardict)
                        pid = os.fork()
                        if pid:
                            parent = True
                            pids.append(pid)
                            workdirs.append( self.workdir)
                            results_files.append(self.results_file)
                        else:
                            parent = False
                            child( self )
                            os._exit( 0 )
        
        # Clean parent
        self.workdir = saved_workdir

        #for par_set in par_sets:
        #    pardict = dict(zip(self.get_par_names(), par_set ) )
        #    self.set_par_values(par_set)
        #    if not self.workdir_base is None:
        #        child_dir = self.workdir_base + '.' + str( index )
        #    jobs.append(job_server.submit(child,(self.model,templatedir,child_dir,), (),("os","subprocess",)))
        #     
        #    print "Job ", str(index), " added to queue"
        #    index += 1
 
        ## Wait for jobs and collect results
        #responses = []
        #for job in jobs:
        #    child_dir, index = job()
        #    print "Job ", str(index), " finished"
        #    self.read_model_files( workdir=str(child_dir) )
        #    if not save:
        #        rmtree( child_dir )
        #    responses.append(self.get_sims())
        #    stdout.flush()

        #responses, samples, status = parallel(self, ncpus, par_sets, templatedir=templatedir,
        #                    workdir_base=workdir_base, save=save)
        #if status:
        #    return 0, 0
        #else:
        return responses, par_sets   
    def get_parstudy(self, outfile=None, *args, **kwargs):
        """ Generate parameter study samples
        
            Parameter
            ---------
            outfile : string
                name of file to output samples to
                If outfile=None, no file is written.
            *args : tuple, list, or ndarray of number of values for each parameter
                    The order is expected to match that produced by prob.par
            
            Returns
            -------
            samples : ndarray 
                Parameter samples
          
        """

        if len(args) > 0 and len(kwargs) > 0:
            print "Warning: dictionary arg will overide keyword args"
            if len(args[0]) > 0:
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
                    self.par[k].nvals = v


        x = []
        for p in self.parlist:
            if p.nvals == 1:
                x.append(numpy.linspace(p.value, p.max, p.nvals))
            if p.nvals > 1:
                x.append(numpy.linspace(p.min, p.max, p.nvals))

        x = list(itertools.product(*x))
        x = numpy.array(x)

        if outfile:
            f = open(outfile, 'w')
            for nm in self.get_par_names():
                f.write(" ")
                f.write("%16s" % nm )
            f.write('\n')
            numpy.savetxt(f, x, fmt='%16lf')

        return x

