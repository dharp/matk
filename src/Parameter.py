class Parameter:
    def __init__(self):
        self.__name = ''
        self.__min = 0
        self.__max = 0
        self.__trans = ''
        self.__initial_value = []
        self.__value = []
        self.__scale = []
        self.__offset = []
        self.__pargrp = ''
        self.__parchglim = ''
    @property
    def name(self):
        return self.__name
    @name.setter
    def name(self,value):
        self.__name = value
    @property
    def min(self):
        return self.__min
    @min.setter
    def min(self,value):
        self.__min = value
    @property
    def max(self):
        return self.__max
    @max.setter
    def max(self,value):
        self.__max = value
    @property
    def trans(self):
        return self.__trans
    @trans.setter
    def trans(self,value):
        self.__trans = value
    @property
    def initial_value(self):
        return self.__initial_value
    @initial_value.setter
    def initial_value(self,value):
        self.__initial_value = value
    @property
    def value(self):
        return self.__value
    @value.setter
    def value(self,value):
        self.__value = value
    @property
    def scale(self):
        return self.__scale
    @scale.setter
    def scale(self,value):
        self.__scale = value
    @property
    def offset(self):
        return self.__offset
    @offset.setter
    def offset(self,value):
        self.__offset = value
    @property
    def pargp(self):
        return self.__pargp
    @pargp.setter
    def pargp(self,value):
        self.__pargp = value
    @property
    def parchglim(self):
        return self.__parchglim
    @parchglim.setter
    def parchglim(self,value):
        self.__parchglim = value
        