from observation import *

class obsgrp(object):
    def __init__(self, name):
        self.name = name
        self.observation = []
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
    def addobservation(self, name, value, **kwargs):
        """Add a parameter to the problem
        
            [-] - optional parameters
            problem.addparameter( name, value, weight, obsgrpnm)
        """
        #mypar = parameter(name,value, **kwargs)
        self.observation.append(observation(name,value,**kwargs))
        