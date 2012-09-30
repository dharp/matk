class Problem:
    def __init__(self):
        self.__npar = 0
        self.__npargp = 0
        self.__nobs = 0
        self.__nobsgp = 0
    @property
    def npar(self):
        return self.__npar
    @npar.setter
    def npar(self,value):
        self.__npar = value
    @property
    def npargp(self):
        return self.__npargp
    @npargp.setter
    def npargp(self,value):
        self.__npargp = value
    @property
    def nobs(self):
        return self.__nobs
    @nobs.setter
    def nobs(self,value):
        self.__nobs = value
    @property
    def nobsgp(self):
        return self.__nobsgp
    @nobsgp.setter
    def nobsgp(self,value):
        self.__nobsgp = value 