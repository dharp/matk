import numpy
from lmfit.parameter import Parameter as LMFitParameter
import copy_reg, new
import platform

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
    def __init__(self, name, value=None, vary=True, min=None, max=None, expr=None, nominal=None, discrete_vals=[], **kwargs):
        if expr is not None and platform.system() is 'Windows':
            raise InputError('expr option not supported on Windows, similar functionality can be achieved using expressions in model functions')
        if nominal is not None and value is None: value=nominal
        LMFitParameter.__init__(self, name=name, value=value, vary=vary, min=min, max=max, expr=expr)
        self.from_internal = self._nobound
        # Discrete parameter
        if len(discrete_vals):
            discrete_vals = numpy.array(discrete_vals)
            # Ensure probabilities sum to one
            discrete_vals[1] /= numpy.sum(discrete_vals[1])
            if min is not None: "WARNING: 'min' option will be ignored for discrete parameter"
            if max is not None: "WARNING: 'max' option will be ignored for discrete parameter"
            if expr is not None: "WARNING: 'expr' option will be ignored for discrete parameter"
            if nominal is not None: "WARNING: 'nominal' option will be ignored for discrete parameter"
            self.min = numpy.min(discrete_vals[0])
            self.max = numpy.max(discrete_vals[0])
            self.dist = 'discrete'
            if value is None:
                self._val = discrete_vals[0][len(discrete_vals[0])/2] 
            elif value not in self.discrete_vals[0]:
                print "ERROR: value is not one of the values in discrete_vals"
            else:
                self._val = value
            self._discrete_vals = discrete_vals
        # Else continuous parameter
        else:
            self.mean = None
            self.std = None
            self._dist = 'uniform'
            self._dist_pars = None
            self._parent = None
            self._nominal = nominal
            for k,v in kwargs.iteritems():
                if k == 'mean':
                    self.mean = float(v)
                elif k == 'std':
                    self.std = float(v)
                elif k == 'dist':
                    self.dist = v
                elif k == 'dist_pars':
                    self.dist_pars = v
                elif k == 'parent':
                    self._parent = v
                else:
                    print k + ' is not a valid argument'
            if self.dist == 'uniform':
                if value is None:
                    if self.max is not None and self.min is not None:
                        self._val = (self.max + self.min)/2.
                    elif self.max is not None:
                        self._val = self.max
                    elif self.min is not None:
                        self._val = self.min
                    else:
                        self._val = 0
                if self.dist_pars is None:
                    # Set lower bound
                    if self.min is not None: mn = self.min
                    else: mn = numpy.nan_to_num(-numpy.inf)
                    # Set range
                    if self.min is not None and self.max is not None: rng = self.max - self.min
                    else: rng = numpy.nan_to_num(numpy.inf)
                    self.dist_pars = (mn,rng)
            elif self.dist == 'norm':
                if self.mean is None: self.mean = 0.
                if self.std is None: self.std = 1.
                self.dist_pars = (self.mean, self.std)
            # Set vary to False if max == min
            if (self.min == self.max) and (self.min is not None) and (self.max is not None): self.vary = False
    def __getstate__(self):
        odict = self.__dict__.copy()
        return odict
    def __setstate__(self,state):
        self.__dict__.update(state)
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
        if self.dist == 'discrete':
            pass
        else:
            self._val = value
            if self._parent:
                self._parent._current = False
    @property
    def nominal(self):
        """ Nominal parameter value, used in info gap decision analyses
        """
        return self._nominal
    @nominal.setter
    def nominal(self,value):
        self._nominal = value
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
        e.g. if dist == uniform, dist_pars = (min,max-min)
        if dist == norm, dist_pars = (mean,stdev))
        """
        return self._dist_pars
    @dist_pars.setter
    def dist_pars(self,value):
        self._dist_pars = value              
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

class Error(Exception):
    pass
             
class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
 



