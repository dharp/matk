class observation(object):
    """ pymads observation class
    """
    def __init__(self, name, value, **kwargs):
        self.name = name
        self.value = value
        self.weight = 1.0
        self.obsgrpnm = 'default'
        for k,v in kwargs.iteritems():
            if k == 'weight':
                self.weight = float(v)
            elif k == 'obsgrpnm':
                self.obsgrpnm = v
            else:
                print k + ' is not a valid argument'
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
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
        