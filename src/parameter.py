import numpy

class Parameter(object):
    """ MATK parameter class
    """
    def __init__(self, name, **kwargs):
        self._name = name
        self._valuelist = []
        self._value = None
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
        self._nvals = 2
        self._vary = True
        self._expr = None
        for k,v in kwargs.iteritems():
            if k == 'value':
                self.value = float(v)
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
            else:
                print k + ' is not a valid argument'
        # If min and max are set, but dist is not, set to uniform
        if not self.max is None and not self.min is None and self.dist is '': self.dist = 'uniform'
        if self.dist == 'uniform':
            if self.min is None or self.max is None: 
                print "Error: Max and min parameter value must be specified for uniform distribution"
                return
            if not self.value is None:
                if not self.min <= self.value <= self.max:
                    print "Error: Value is not within min and max values"
                    return
            else:
                self.value = (self.max + self.min)/2.
            range = self.max - self.min
            self.dist_pars = (self.min, range)
        elif self.dist == 'norm':
            if self.mean is None or self.std is None:
                print "Error: Mean and std. dev. required for normal distribution"
            else:
                self.dist_pars = (self.mean, self.std)
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
        return self._value
    @value.setter
    def value(self,value):
        self._value = value
        if not self.value is None:
            self._valuelist.append(self.value)
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
    @property
    def calib_value(self):
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

        This code borrows heavily from lmfit, which borrows heavily from
        JJ Helmus' leastsqbound.py
        """
        try:
            self._min
            self._max
        except NameError: pass
        else:
            if self._min in (None, -numpy.inf) and self._max in (None, numpy.inf):
                self._func_value = lambda val: val
                self._calib_value =  self._value 
            elif self._max in (None, numpy.inf):
                self._func_value = lambda val: numpy.sqrt((val - self.min + 1)**2 - 1)
                self._calib_value = self._min - 1 + numpy.sqrt(self._value*self._value + 1)
            elif self._min in (None, -numpy.inf):
                self._func_value = lambda val: numpy.sqrt((self.max - val + 1)**2 - 1)
                self._calib_value = self._max + 1 - numpy.sqrt(self._value*self._value + 1)
            else:
                self._func_value = lambda val: numpy.arcsin(2*(val - self.min)/(self.max - self.min) - 1)
                self._calib_value = self._min + (numpy.sin(self._value) + 1) * (self._max - self._min) / 2 

            return self._calib_value 
             


