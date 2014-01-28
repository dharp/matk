import numpy
from lmfit.parameter import Parameter as LMFitParameter
import copy_reg, new

# Use copy_reg to allow pickling of bound methods
def make_instancemethod(inst, methodname):
    return getattr(inst, methodname)
def pickle_instancemethod(method):
    return make_instancemethod, (method.im_self, method.im_func.__name__)
copy_reg.pickle(new.instancemethod, pickle_instancemethod,
make_instancemethod)

class Parameter(LMFitParameter):
    """ MATK parameter class
    """
    def __init__(self, name, value=None, vary=True, min=None, max=None, expr=None, discrete_vals=[], discrete_counts=[], **kwargs):
        LMFitParameter.__init__(self, name=name, value=value, vary=vary, min=min, max=max, expr=expr)
        if len(discrete_counts) and (len(discrete_counts) != len(discrete_vals)):
            print "ERROR: discrete_counts requires equal number of discrete_vals"
            return
        elif (min or max) and len(discrete_vals):
            print "ERROR: discrete_vals cannot be set with min or max"
            return
        elif len(discrete_vals) and not len(discrete_counts):
            self._discrete_vals = numpy.array(discrete_vals)
            self._discrete_counts = numpy.ones(len(discrete_vals))
        elif len(discrete_vals) and len(discrete_counts):
            self._discrete_vals = numpy.array(discrete_vals)
            self._discrete_counts = numpy.array(discrete_counts)
        else:
            self._discrete_vals = discrete_vals
            self._discrete_counts = discrete_counts
        self._mean = None
        self._std = None
        self._dist = ''
        self._dist_pars = ()
        self._nvals = 2
        self._parent = None
        for k,v in kwargs.iteritems():
            if k == 'mean':
                self.mean = float(v)
            elif k == 'std':
                self.std = float(v)
            elif k == 'dist':
                self.dist = v
            elif k == 'dist_pars':
                self.dist_pars = v
            elif k == 'nvals':
                self.nvals = v
            elif k == 'parent':
                self._parent = v
            else:
                print k + ' is not a valid argument'
        # If min and max are set, but dist is not, set to uniform
        if not self.max is None and not self.min is None and self.dist is '': self.dist = 'uniform'
        if self.dist == 'uniform':
            if self.min is None or self.max is None: 
                print "Error: Max and min parameter value must be specified for uniform distribution"
                return
            if not self._val is None:
                if not self.min <= self._val <= self.max:
                    print "Error: Value is not within min and max values"
                    return
            else:
                self._val = (self.max + self.min)/2.
            range = self.max - self.min
            self.dist_pars = (self.min, range)
        elif self.dist == 'norm':
            if self.mean is None or self.std is None:
                print "Error: Mean and std. dev. required for normal distribution"
            else:
                self.dist_pars = (self.mean, self.std)
        if self.expr is not None and self.dist is '':
            self.dist='uniform'
    def __getstate__(self):
        odict = self.__dict__.copy()
        return odict
    def __setstate__(self,state):
        self.__dict__.update(state)
    @property
    def mean(self):
        """ Parameter mean
        """
        return self._mean
    @mean.setter
    def mean(self,value):
        self._mean = value
    @property
    def std(self):
        """ Parameter st. dev.
        """
        return self._std
    @std.setter
    def std(self,value):
        self._std = value
    @property
    def value(self):
        """ Parameter value
        """
        if not self.expr is None:
            return self._getval()
        else:
            return self._val
    @value.setter
    def value(self,value):
        if len(self._discrete_vals):
            pass
        self._val = value
        if self._parent:
            self._parent._current = False
    @property
    def dist(self):
        """ Probabilistic distribution of parameter belonging to scipy.stats
        module
        """
        return self._dist
    @dist.setter
    def dist(self,value):
        self._dist = value        
    @property
    def dist_pars(self):
        """ Distribution parameters required by self.dist 
        (e.g. if dist == uniform, dist_pars = (min,max-min))
        """
        return self._dist_pars
    @dist_pars.setter
    def dist_pars(self,value):
        self._dist_pars = value              
    @property
    def nvals(self):
        """ Number of values the paramter will take for parameter studies
        """
        return self._nvals
    @nvals.setter
    def nvals(self,value):
        self._nvals = value              
    @property
    def vary(self):
        """ Boolean indicating whether or not to vary parameter
        """
        return self._vary
    @vary.setter
    def vary(self,value):
        self._vary = value              
    @property
    def expr(self):
        """ Mathematical expression to use to evaluate value
        """
        return self._expr
    @expr.setter
    def expr(self,value):
        self._expr = value              
    def _nobound(self,val):
        return val
    def _minbound(self,val):
        return self.min - 1 + numpy.sqrt(val*val + 1)
    def _maxbound(self,val):
        return self.max + 1 - numpy.sqrt(val*val + 1)
    def _bound(self,val):
        return self.min + (numpy.sin(val) + 1) * (self.max - self.min) / 2
    def setup_bounds(self):
        """set up Minuit-style internal/external parameter transformation
        of min/max bounds.

        returns internal value for parameter from self.value (which holds
        the external, user-expected value).   This internal values should
        actually be used in a fit....

        As a side-effect, this also defines the self.from_internal method
        used to re-calculate self.value from the internal value, applying
        the inverse Minuit-style transformation.  This method should be
        called prior to passing a Parameter to the user-defined objective
        function.

        This code borrows heavily from JJ Helmus' leastsqbound.py
        """
        if self.min in (None, -numpy.inf) and self.max in (None, numpy.inf):
            #self.from_internal = lambda val: val
            self.from_internal = self._nobound
            _val  = self._val
        elif self.max in (None, numpy.inf):
            #self.from_internal = lambda val: self.min - 1 + sqrt(val*val + 1)
            self.from_internal = self._minbound
            _val  = numpy.sqrt((self._val - self.min + 1)**2 - 1)
        elif self.min in (None, -numpy.inf):
            #self.from_internal = lambda val: self.max + 1 - sqrt(val*val + 1)
            self.from_internal = self._maxbound
            _val  = numpy.sqrt((self.max - self._val + 1)**2 - 1)
        else:
            #self.from_internal = lambda val: self.min + (sin(val) + 1) * \
            #                     (self.max - self.min) / 2
            self.from_internal = self._bound 
            _val  = numpy.arcsin(2*(self._val - self.min)/(self.max - self.min) - 1)
        return _val

             


