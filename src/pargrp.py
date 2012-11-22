from parameter import Parameter

class ParameterGroup(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.derinc = None
        self.derinclb = None
        self.derincmul = None
        self.derincmthd = None
        for k,v in kwargs.iteritems():
            if k == 'derinc':
                self.derinc = v
            elif k == 'derinclb':
                self.derinclb = v
            elif k == 'derincmul':
                self.derincmul = v
            elif k == 'derincmthd':
                self.derincmthd = v
            else:
                print k + ' is not a valid argument'
        self.parameter = []
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
    @property
    def derinc(self):
        return self._derinc
    @derinc.setter
    def derinc(self,value):
        self._derinc = value
    @property
    def derinclb(self):
        return self._derinclb
    @derinclb.setter
    def derinclb(self,value):
        self._derinclb = value
    @property
    def derincmul(self):
        return self._derincmul
    @derincmul.setter
    def derincmul(self,value):
        self._derincmul = value
    @property
    def derincmthd(self):
        return self._derincmth
    @derincmthd.setter
    def derincmthd(self,value):
        self._derincmth = value
    def add_parameter(self, name, initial_value, **kwargs):
        """Add a parameter to the problem
        
            [-] - optional parameters
            problem.add_parameter( name, initial_value, min=0.0, 
            max=1.0, offset=0.0, scale=1.0, trans=None, parchglim=None)
        """
        self.parameter.append(Parameter(name,initial_value,**kwargs))
        
    def __iter__(self):
        return iter(self.parameter)