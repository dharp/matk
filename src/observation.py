class Observation(object):
    """ pymads observation class
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.value = None
        self.sim = None
        self.residual = None
        for k,v in kwargs.iteritems():
            if k == 'weight':
                self.weight = float(v)
            elif k == 'value':
                self.value = float(v)
            elif k == 'sim':
                self.sim = float(v)
            else:
                print k + ' is not a valid argument'
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self,value):
        self._value = value
    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self,value):
        self._weight = value
    @property
    def sim(self):
        return self._sim
    @sim.setter
    def sim(self,value):
        if value is not None:
            self._sim = float(value)
        else:
            self._sim = None
    @property
    def residual(self):
        self.residual = self.value - self.sim
        return self._residual
    @residual.setter
    def residual(self,value):
        self._residual = value
