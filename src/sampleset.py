import sys
import numpy
import string
from scipy import stats
from shutil import rmtree
try:
    from matplotlib import pyplot as plt
    plotflag = True
except ImportError as exc:
    sys.stderr.write("Warning: failed to import matplotlib module. Plots will not be produced. ({})".format(exc))
    plotflag = False

class SampleSet(object):
    """ MATK SampleSet class - Stores information related to a sample
        including parameter samples, associated responses, and sample indices
    """
    def __init__(self,name,samples,index_start=1,**kwargs):
        self.name = name
        #self._samples = samples
        self._responses = None
        self._indices = None
        parnames = None
        self._obsnames = None
        self._index_start = index_start
        self._parent = None
        for k,v in kwargs.iteritems():
            if k == 'responses':
                if not v is None:
                    if isinstance( v, (list,numpy.ndarray)):
                        self.responses = v
                    else:
                        print "Error: Responses are not a list or ndarray"
                        return
            elif k == 'indices':
                if not v is None:
                    if isinstance( v, (list,numpy.ndarray)):
                        self.indices = v
                    else:
                        print "Error: Indices are not a list or ndarray"
                        return
            elif k == 'parnames':
                if not v is None:
                    if isinstance( v, (tuple,list,numpy.ndarray)):
                        parnames = v
                    else:
                        print "Error: Parnames are not a tuple, list or ndarray"
                        return
            elif k == 'obsnames':
                if not v is None:
                    if isinstance( v, (tuple,list,numpy.ndarray)):
                        self._obsnames = v
                    else:
                        print "Error: Obsnames are not a tuple, list or ndarray"
                        return
            elif k == 'parent':
                self._parent = v
            else:
                print k + ' is not a valid argument'
        self.samples = Samples(samples,parnames,self) 
        # Set default indices if None
        if self.indices is None and not self.samples.values is None:
            if not self.index_start is None:
                self.indices = numpy.arange(index_start,index_start+self.samples.values.shape[0])
            else:
                self.indices = numpy.arange(self.samples.values.shape[0])+1
    @property
    def name(self):
        """Sample set name
        """
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
    @property
    def responses(self):
        """Ndarray of sample set responses, rows are samples, columns are responses associated with observations in order of MATKobject.obslist
        """
        return self._responses
    @responses.setter
    def responses(self,value):
        if self.samples is None and value is None:
            self._responses = value
        elif self.samples is None and not value is None:
            print "Error: Samples are not defined"
            return
        elif value is None:
            self._responses = value
        elif isinstance(value, numpy.ndarray):
            if not value.shape[0] == self.samples.values.shape[0]:
                print "Error: number of reponses does not equal number of samples"
                return
            else:
                self._responses = numpy.array(value)
        elif isinstance(value, list):
            if not len(value) == self.samples.values.shape[0]:
                print "Error: number of responses does not equal number of samples"
                return
            else:
                self._responses = numpy.array(value)
        else:
            print "Error: Responses must be a list or ndarray with nsample rows"
            return
    @property
    def responses_recarray(self):
        """ Structured (record) array of responses
        """
        return numpy.rec.fromarrays(self._responses.T,names=self.obsnames)
    @property
    def indices(self):
        """ Array of sample indices
        """ 
        return self._indices
    @indices.setter
    def indices(self,value):
        if self.samples is None and value is None:
            self._indices = value
        elif self.samples is None and not value is None:
            print "Error: Samples are not defined"
            return
        elif value is None:
            self._indices = value
        elif not len(value) == self.samples.values.shape[0]:
            print "Error: number of indices does not equal number of samples"
            return
        else:
            self._indices = value
    @property
    def obsnames(self):
        """ Array of observation names
        """ 
        if not self._parent is None:
            if len(self._parent.obsnames):
                self._obsnames = self._parent.obsnames
        return self._obsnames
    @property
    def index_start(self):
        """ Starting integer value for sample indices
        """
        return self._index_start
    @index_start.setter
    def index_start(self,value):
        if not isinstance( value, int):
            print "Error: Expecting integer"
            return
        self._index_start = value
        if not self.samples is None:
            self.indices = numpy.arange(self.index_start,self.index_start+self.samples.values.shape[0])
    def corr(self, type='pearson', plot=False):
        """ Calculate correlation coefficients of parameters and responses

            :param type: Type of correlation coefficient (pearson by default, spearman also avaialable)
            :type type: str
            :returns: ndarray(fl64) -- Correlation coefficients
        """
        corrlist = []
        if type is 'pearson':
            for i in range(self.samples.values.shape[1]):
                corrlist.append([stats.pearsonr(self.samples[:,i],self.responses[:,j])[0] for j in range(self.responses.shape[1])])
        elif type is 'spearman':
            for i in range(self.samples.values.shape[1]):
                corrlist.append([stats.spearmanr(self.samples[:,i],self.responses[:,j])[0] for j in range(self.responses.shape[1])])
        else:
            print "Error: type not recognized"
            return
        corrcoef = numpy.array(corrlist)
        # Print 
        dum = ' '
        print string.rjust(`dum`, 8),
        for nm in self.obsnames:
            print string.rjust(`nm`, 20),
        print ''
        for i in range(corrcoef.shape[0]):
            print string.ljust(`self.samples.parnames[i]`, 8),
            for c in corrcoef[i]:
                print string.rjust(`c`, 20),
            print ''

        if plot and plotflag:
            # Plot
            plt.pcolor(corrcoef, vmin=-1, vmax=1)
            plt.colorbar()
            if type is 'pearson':
                plt.title('Pearson Correlation Coefficients')
            elif type is 'spearman':
                plt.title('Spearman Rank Correlation Coefficients')
            plt.yticks(numpy.arange(0.5,len(self.samples.parnames)+0.5),self.samples.parnames)
            plt.xticks(numpy.arange(0.5,len(self.obsnames)+0.5),self.obsnames)
            plt.show()

        return corrcoef
    def run(self, ncpus=1, templatedir=None, workdir_base=None,
                    save=True, reuse_dirs=False, outfile=None, logfile=None, verbose=True ):
        """ Run model using values in samples for parameter values
            If samples are not specified, LHS samples are produced
            
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
            :param outfile: File to write results to
            :type outfile: str
            :param logfile: File to write details of run to during execution
            :type logfile: str
            :returns: tuple(ndarray(fl64),ndarray(fl64)) - (Matrix of responses from sampled model runs siz rows by npar columns, Parameter samples, same as input samples if provided)
        """
        if templatedir:
            self._parent.templatedir = templatedir
        if workdir_base:
            self._parent.workdir_base = workdir_base
                
        if ncpus > 0:
            out, samples = self._parent.parallel(ncpus, self.samples.values,
                 indices=self.indices, templatedir=templatedir, workdir_base=workdir_base, 
                 save=save, reuse_dirs=reuse_dirs, verbose=verbose, logfile=logfile)
        else:
            print 'Error: number of cpus (ncpus) must be greater than zero'
            return
        out = numpy.array(out)
        self.responses = out 
        self._obsnames = self._parent.obsnames
        if not outfile is None:
            self.savetxt( outfile )

        return out
    def savetxt( self, outfile):
        ''' Save sampleset to file

            :param outfile: Name of file where sampleset will be written
            :type outfile: str
        '''

        x = numpy.column_stack([self.indices,self.samples.values])
        if not self.responses is None:
            x = numpy.column_stack([x,self.responses])

        if outfile:
            f = open(outfile, 'w')
            f.write("%-8s" % 'index' )
            # Print par names
            for nm in self.samples.parnames:
                f.write(" %16s" % nm )
            # Print obs names if responses exist
            if not self.responses is None:
                if len(self.obsnames) == 0:
                    for i in range(self.responses.shape[1]):
                        f.write("%16s" % 'obs'+str(i+1) )
                else:
                    for nm in self.obsnames:
                        f.write(" %16s" % nm )
            f.write('\n')
            for row in x:
                if isinstance( row[0], str ):
                    f.write("%-8s" % row[0] )
                else:
                    f.write("%-8d" % row[0] )
                for i in range(1,len(row)):
                    if isinstance( row[i], str):
                        f.write(" %16s" % row[i] )
                    else:
                        f.write(" %16lf" % row[i] )
                f.write('\n')
            f.close()
            
class Samples(object):
    """ MATK samples class - Stores information related to a sample
        includeing parameter samples, associated responses, and sample indices
    """
    def __init__(self,samples,parnames,parent):
        self._values = samples
        self._parnames = parnames
        self._parent = parent
    @property
    def parnames(self):
        """ Array of parameter names
        """ 
        if not self._parent._parent is None:
            if len(self._parent._parent.parnames):
                self._parnames = self._parent._parent.parnames
        return self._parnames
    @property
    def values(self):
        """Ndarray of parameter samples, rows are samples, columns are parameters in order of MATKobject.parlist
        """
        return self._values
    @values.setter
    def values(self,value):
        if not isinstance( value, (list,numpy.ndarray)):
            print "Error: Parameter samples are not a list or ndarray"
            return
        # If list, convert to ndarray
        if isinstance( value, list ):
            self._values = numpy.array(value)
        else:
            self._values = value
    @property
    def recarray(self):
        """ Structured (record) array of samples
        """
        return numpy.rec.fromarrays(self._values.T,names=self._parnames)
 




