import numpy
import string
from scipy import stats
from matplotlib import pyplot as plt

class SampleSet(object):
    """ MATK samples class - Stores information related to a sample
        includeing parameter samples, associated responses, and sample indices
    """
    def __init__(self, name, samples, index_start=1, **kwargs):
        self.name = name
        self._samples = samples
        self._responses = None
        self._indices = None
        self._parnames = None
        self._obsnames = None
        self._index_start = index_start
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
                        self.parnames = v
                    else:
                        print "Error: Parnames are not a tuple, list or ndarray"
                        return
            elif k == 'obsnames':
                if not v is None:
                    if isinstance( v, (tuple,list,numpy.ndarray)):
                        self.obsnames = v
                    else:
                        print "Error: Obsnames are not a tuple, list or ndarray"
                        return
            else:
                print k + ' is not a valid argument'
        # Set default indices if None
        if self.indices is None and not self.samples is None:
            if not self.index_start is None:
                self.indices = numpy.arange(index_start,index_start+self.samples.shape[0])
            else:
                self.indices = numpy.arange(self.samples.shape[0])+1
    @property
    def name(self):
        """Sample set name
        """
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
    @property
    def samples(self):
        """Ndarray of parameter samples, rows are samples, columns are parameters in order of MATKobject.parlist
        """
        return self._samples
    @samples.setter
    def samples(self,value):
        if not isinstance( value, (list,numpy.ndarray)):
            print "Error: Parameter samples are not a list or ndarray"
            return
        # If list, convert to ndarray
        if isinstance( value, list ):
            self._samples = numpy.array(value)
        else:
            self._samples = value
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
            if not value.shape[0] == self.samples.shape[0]:
                print "Error: number of indices does not equal number of samples"
                return
            else:
                self._responses = numpy.array(value)
        elif isinstance(value, list):
            if not len(value) == self.samples.shape[0]:
                print "Error: number of indices does not equal number of samples"
                return
            else:
                self._responses = numpy.array(value)
        else:
            print "Error: Responses must be a list or ndarray with nsample rows"
            return
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
        elif not len(value) == self.samples.shape[0]:
            print "Error: number of indices does not equal number of samples"
            return
        else:
            self._indices = value
    @property
    def parnames(self):
        """ Array of parameter names
        """ 
        return self._parnames
    @parnames.setter
    def parnames(self,value):
        if self.samples is None and value is None:
            self._parnames = value
        elif self.samples is None and not value is None:
            print "Error: Samples are not defined"
            return
        elif value is None:
            self._parnames = value
        elif not len(value) == self.samples.shape[1]:
            print "Error: number of parnames does not equal number of samples"
            return
        else:
            self._parnames = value
    @property
    def obsnames(self):
        """ Array of observation names
        """ 
        return self._obsnames
    @obsnames.setter
    def obsnames(self,value):
        if self.samples is None and value is None:
            self._obsnames = value
        elif self.samples is None and not value is None:
            print "Error: Samples are not defined"
            return
        elif value is None:
            self._obsnames = value
        else:
            self._obsnames = value
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
            self.indices = numpy.arange(self.index_start,self.index_start+self.samples.shape[0])
    def corr(self, type='pearson', plot=False):
        """ Calculate correlation coefficients of parameters and responses

            :param type: Type of correlation coefficient (pearson by default, spearman also avaialable)
            :type type: str
            :returns: ndarray(fl64) -- Correlation coefficients
        """
        corrlist = []
        if type is 'pearson':
            for i in range(self.samples.shape[1]):
                corrlist.append([stats.pearsonr(self.samples[:,i],self.responses[:,j])[0] for j in range(self.responses.shape[1])])
        elif type is 'spearman':
            for i in range(self.samples.shape[1]):
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
            print string.ljust(`self.parnames[i]`, 8),
            for c in corrcoef[i]:
                print string.rjust(`c`, 20),
            print ''

        if plot:
            # Plot
            plt.pcolor(corrcoef)
            plt.colorbar()
            plt.yticks(numpy.arange(0.5,len(self.parnames)+0.5),self.parnames)
            plt.xticks(numpy.arange(0.5,len(self.obsnames)+0.5),self.obsnames)
            plt.show()


        return corrcoef
            





