import numpy

class ModelInstruction(object):
    """pymads Templateate file class
    """
    def __init__(self,insflname,modelflname):
        self.insflname = insflname
        self.modelflname = modelflname
        f = open( self.insflname, 'r')
        self.lines = f.readlines()
        lines = numpy.array(self.lines)
        values = self.lines[0].split()
        self.lines = lines[1:]
        if values[0] != 'pif':
            print "%s doesn't appear to be a PEST instruction file" % self.insflname
            return 0
        self.marker = values[1]
    @property
    def insflname(self):
        return self._insflname
    @insflname.setter
    def insflname(self,value):
        self._insflname = value
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
        