class Parameter(object):
    """ pymads parameter class
    """
    def __init__(self, name, initial_value, **kwargs):
        self.name = name
        self.initial_value = initial_value
        self.value = []
        self.value.append( initial_value)
        self.min = None
        self.max = None
        self.trans = 'none'
        self.scale = 1.0
        self.offset = 0.0
        self.parchglim = None
        self.pargrpnm = 'default'
        for k,v in kwargs.iteritems():
            if k == 'min':
                self.min = float(v)
            elif k == 'max':
                self.max = float(v)
            elif k == 'trans':
                self.trans = v
            elif k == 'scale':
                self.scale = float(v)
            elif k == 'offset':
                self.offset = float(v)
            elif k == 'parchglim':
                self.parchglim = v
            elif k == 'pargrpnm':
                self.pargrpnm = v
            else:
                print k + ' is not a valid argument'
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
    @property
    def min(self):
        return self._min
    @min.setter
    def min(self,value):
        self._min = value
    @property
    def max(self):
        return self._max
    @max.setter
    def max(self,value):
        self._max = value
    @property
    def trans(self):
        return self._trans
    @trans.setter
    def trans(self,value):
        self._trans = value
    @property
    def initial_value(self):
        return self._initial_value
    @initial_value.setter
    def initial_value(self,value):
        self._initial_value = value
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self,value):
        self._value = value
    @property
    def scale(self):
        return self._scale
    @scale.setter
    def scale(self,value):
        self._scale = value
    @property
    def offset(self):
        return self._offset
    @offset.setter
    def offset(self,value):
        self._offset = value
    @property
    def pargrpnm(self):
        return self._pargrpnm
    @pargrpnm.setter
    def pargrpnm(self,value):
        self._pargrpnm = value
    @property
    def parchglim(self):
        return self._parchglim
    @parchglim.setter
    def parchglim(self,value):
        self._parchglim = value
        