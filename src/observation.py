class Observation(object):
    """ pymads observation class
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.weight = 1.0
        self.obsgrpnm = 'default'
        for k,v in kwargs.iteritems():
            if k == 'weight':
                self.weight = float(v)
            elif k == 'obsgrpnm':
                self.obsgrpnm = v
            elif k == 'value':
                self.value = float(v)
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
    def obsgrpnm(self):
        return self._obsgrpnm
    @obsgrpnm.setter
    def obsgrpnm(self,value):
        self._obsgrpnm = value
    @property
    def sim_value(self):
        return self._sim_value
    @sim_value.setter
    def sim_value(self,value):
        self._sim_value = float(value)
    @property
    def residual(self):
        return self._residual
    @residual.setter
    def residual(self,value):
        self._residual = value
