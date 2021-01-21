import numpy as np

class StatusFF(object):
    """
    指示y_i是否为c
    """
    def __init__ (self, c):
        self.c = c

    def __call__ (self, y_, y, X, i):
        return 1 if y == self.c else 0

class ObsFF(object):
    """
    指示是否y_i=c，x_{ij}=d
    """
    def __init__ (self, j, c, d):
        self.j, self.c, self.d = j, c, d

    def __call__ (self, y_, y, X, i):
        return 1 if y==self.c and X[i, self.j] == self.d else 0

class TransFF(object):
    """
    指示是否y_{i}=c, y_{i-1}=d
    """
    def __init__ (self, c, d):
        self.c, self.d = c, d
    
    def __call__ (self, y_, y, X, i):
        return 1 if y_ == self.d and y == self.c else 0
    