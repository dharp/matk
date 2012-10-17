import numpy

class ModelTemplate(object):
    """pymads Templateate file class
    """
    def __init__(self,tplflname,modelflname):
        self.tplflname = tplflname
        self.modelflname = modelflname
        f = open( self.tplflname, 'r')
        self.lines = f.readlines()
        lines = numpy.array(self.lines)
        values = self.lines[0].split()
        self.lines = lines[1:]
        if values[0] != 'ptf':
            print "%s doesn't appear to be a PEST template file" % self.tplflname
            return 0
        self.marker = values[1]
    @property
    def tplflname(self):
        return self._tplflname
    @tplflname.setter
    def tplflname(self,value):
        self._tplflname = value
    @property
    def modelflname(self):
        return self._modelflname
    @modelflname.setter
    def modelflname(self,value):
        self._modelflname = value
    @property
    def marker(self):
        return self._marker
    @marker.setter
    def marker(self,value):
        self._marker = value 
        