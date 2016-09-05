import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import bisection as bis
class Newton():
    
    def descent(self, alpha = 1, X = np.array([[0.0],[0.0]]), precision = 10**-6):
        self.Xs = []       # Candidates
        self.Xs.append(X)  # First Candidate (x0)
        self.precision = precision
        H = self.modify(self.hessiana())    # Initial hessiana
        d = - np.dot(la.inv(H), self.gradF(X))  # Initial direction
        self.alpha = bis.getAlpha(self.Xs[-1], d)

        while la.norm(d) > self.precision:
            X = X + alpha*d
            self.Xs.append(X)
            X = self.Xs[-1]
            H = self.modify(self.hessiana())
            d = - np.dot(la.inv(H), self.gradF(X))

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
        x1 = X[0][0]
        x2 = X[1][0]

        return np.array([[4*(x1 - 2)],[4*(x2 - 3)]])

    ''' Second gradient of partial derivatives of f(x)'''
    def hessiana(self):
        hess = np.array([[4, 0], [0, 4]])

        return hess

    def getEigenvalue(self, H):
        eigs = la.eigvals(H)
        return min(eigs)

    ''' ModifiedNewton '''
    def modify(self, H):
        epsilon = 0.01
        e = self.getEigenvalue(H)
        if(e > 0):
            return H
        else:
            M = H + (epsilon - e) * np.identity(2)

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



 
