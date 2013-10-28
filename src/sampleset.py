import numpy

class SampleSet(object):
    """ MATK samples class - Stores information related to a sample
        includeing parameter samples, associated responses, and sample indices
    """
    def __init__(self, name, index_start=1, **kwargs):
        self.name = name
        self._samples = None
        self._responses = None
        self._indices = None
        self._index_start = index_start
        for k,v in kwargs.iteritems():
            if k == 'samples':
                if not v is None:
                    if isinstance( v, list):
                        self.samples = numpy.array(v)
                    elif isinstance( v, numpy.ndarray ):
                        self.samples = v
                    else:
                        print "Error: Samples are not a list or ndarray"
                        return
            elif k == 'responses':
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





