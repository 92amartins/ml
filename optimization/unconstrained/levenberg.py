import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import bisection as bis
class Levenberg():
    
    def descent(self, X = np.array([[0.0],[0.0]]), precision = 10**-6):
        self.Xs = []       # Candidates
        self.Xs.append(X)  # First Candidate (x0)
        self.precision = precision

        while la.norm(self.gradF(self.Xs[-1])) > self.precision:
            X = self.Xs[-1]
            grT = self.buildGradientR().T
            H = self.hessiana(X)
            d = np.dot(np.dot(la.inv(H), grT), self.r(X))
            X = X - 1*d  # TODO: calculate optimal alpha
            self.Xs.append(X)

        return self.Xs[-1]

    def r(self, X):
        x1 = X[0][0]
        x2 = X[1][0]
        
        return np.array([[4*(x1 - 2)], [4*(x2 - 3)]])
    
    ''' Builds the matrix of gradient of r(x)'''
    def buildGradientR(self):   # TODO: Transform function in CONSTANT
        return np.array([[4, 0],
                         [0, 4]])

    def gradF(self, X):
        grT = self.buildGradientR().T
        r = self.r(X)
       
        return np.dot(grT, r)

    ''' Returns Gradient of r(x)T * Gradient of r(x)'''
    def hessiana(self, X):
        grT = self.buildGradientR().T
        gr = self.buildGradientR()
        
        return np.dot(grT, gr)

    def plot(self, title, xlabel, ylabel):
        x1s = []
        x2s = []
        
        for X in self.Xs:
            x1s.append(X[0][0])
            x2s.append(X[1][0])

        plt.plot(x1s, x2s, 'ro')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
