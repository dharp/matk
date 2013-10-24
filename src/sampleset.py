import numpy

class SampleSet(object):
    """ MATK samples class - Stores information related to a sample
        includeing parameter samples, associated responses, and sample indices
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.samples = None
        self.responses = None
        self.indices = None
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
            self.indices = numpy.arange(self.samples.shape[0]) + 1
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
        self._samples = value
    @property
    def responses(self):
        """Ndarray of sample set responses, rows are samples, columns are responses associated with observations in order of MATKobject.obslist
        """
        return self._responses
    @responses.setter
    def responses(self,value):
        self._responses = value
    @property
    def indices(self):
        """ Array of sample indices
        """
        return self._indices
    @indices.setter
    def indices(self,value):
        self._indices = value


