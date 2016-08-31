import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

class Gradient():

    def __init__(self, alpha = 0.25, precision = 10**-6):
        self.prec = precision
        self.alpha = alpha
        
    def descent(self):
        x = [0.0, 0.0]
        Xs = list()

        while True:
            d = np.negative(self.gradf(x))
            if(la.norm(d) <= self.prec):
                break
            x = x + self.alpha * d

        print(x)
        return x

    
    def gradf(self, X):
        return [4*(X[0] - 2),
                4*(X[1] - 3)]
                         
  
    def evaluate(self, X):
        return 2*(X[0] - 2)**2 + 2*(X[1] - 3)**2
         
