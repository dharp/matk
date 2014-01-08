import numpy
from lmfit.parameter import Parameter
import copy_reg, new

# Use copy_reg to allow pickling of bound methods
def make_instancemethod(inst, methodname):
    return getattr(inst, methodname)
def pickle_instancemethod(method):
    return make_instancemethod, (method.im_self, method.im_func.__name__)
copy_reg.pickle(new.instancemethod, pickle_instancemethod,
make_instancemethod)

class Parameter(Parameter):
    """ MATK parameter class
    """
    def __init__(self, name, **kwargs):
        self._name = name
        self._valuelist = []
        self._val = None
        self._min = None
        self._max = None
        self._mean = None
        self._std = None
        self._trans = 'none'
        self._scale = 1.0
        self._offset = 0.0
        self._parchglim = None
        self._pargrpnm = 'default'
        self._dist = ''
        self._dist_pars = ()
        self._nvals = 2
        self._vary = True
        self._expr = None
        self._parent = None
        self.deps   = None
        self.stderr = None
        self.correl = None
        for k,v in kwargs.iteritems():
            if k == 'value':
                self._val = float(v)
            elif k == 'min':
                self.min = float(v)
            elif k == 'max':
                self.max = float(v)
            elif k == 'mean':
                self.mean = float(v)
            elif k == 'std':
                self.std = float(v)
            elif k == 'trans':
                self.trans = v
            elif k == 'scale':
                self.scale = float(v)
            elif k == 'offset':
                self.offset = float(v)
            elif k == 'parchglim':
                self.parchglim = v
            elif k == 'dist':
                self.dist = v
            elif k == 'dist_pars':
                self.dist_pars = v
            elif k == 'nvals':
                self.nvals = v
            elif k == 'vary':
                self.vary = v
            elif k == 'expr':
                self.expr = v
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
        self.user_value = self._val
        self.init_value = self._val
    def __getstate__(self):
        odict = self.__dict__.copy()
        return odict
    def __setstate__(self,state):
        self.__dict__.update(state)
    @property    
    def name(self):
        """ Parameter name
        """
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
    @property
    def min(self):
        """ Parameter lower bound
        """
        return self._min
    @min.setter
    def min(self,value):
        self._min = value
    @property
    def max(self):
        """ Parameter upper bound
        """
        return self._max
    @max.setter
    def max(self,value):
        self._max = value
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
    def trans(self):
        return self._trans
    @trans.setter
    def trans(self,value):
        self._trans = value
    @property
    def value(self):
        """ Parameter value
        """
        return self._getval()
    @value.setter
    def value(self,value):
    #    if not self._min is None and value < self._min:
    #        print "Error: Attempted to set "+self.name+" below min ("+str(self._min)+")"
    #        print self.name+" set to min"
    #        self._value = self._min
    #    elif not self._max is None and value > self._max:
    #        print "Error: Attempted to set "+self.name+" above max ("+str(self._max)+")"
    #        print self.name+" set to max"
    #        self._value = self._max
    #    else:
    #        self._value = value
    #    if not self.value is None:
    #        self._valuelist.append(self.value)
        self._val = value
        if self._parent:
            self._parent._current = False
    @property
    def scale(self):
        """ Scale factor to multiply parameter by
        """
        return self._scale
    @scale.setter
    def scale(self,value):
        self._scale = value
    @property
    def offset(self):
        """ Offset to add to parameter
        """
        return self._offset
    @offset.setter
    def offset(self,value):
        self._offset = value
    @property
    def parchglim(self):
        return self._parchglim
    @parchglim.setter
    def parchglim(self,value):
        self._parchglim = value
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

             


