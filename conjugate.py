import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

'''Conjugate for 2-dimensions minimization'''
class Conjugate():
    
    def __init__(self, precision = 10**-6, X = np.array([[0.] , [0.]])):
         self.Xs = [] # Candidates
         self.Xs.append(X) # Append first candidate (X0)
         self.precision = precision # Stop criteria

    '''Returns the point (x1, x2) which minimizes f'''
    def descent(self):
        d = - self.gradient(self.Xs[0])
        
        # Calculate new X and Update direction
        while la.norm(d) > self.precision:
            # Obtain optimal alpha
            H = self.hessiana()
            alpha = self.alpha(d, H)
            X = self.Xs[-1] + alpha * d
            self.Xs.append(X)

            # Update direction based in Beta
            d = - self.gradient(X) + self.beta((self.gradient(X).T), H, d) * d
        return self.Xs[-1]

    '''Returns the gradient of f(x1, x2) evaluated at X'''
    def gradient(self, X):
        x1 = X[0][0]
        x2 = X[1][0]
        
        return np.array([[4*(x1 - 2)],
                         [4*(x2 - 3)]])

    ''' Calculates the optimal alpha for the iteration'''
    def alpha(self, d, H):
        return np.dot(d.T, d) / np.dot(np.dot(d.T, H), d)

    def beta(self, gT, H, d):
        # Calculates numerator
        betaN = np.dot(gT, H)
        betaN = np.dot(betaN, d)

        # Calculates Denominator
        betaD = np.dot(d.T, H)
        betaD = np.dot(betaD, d)

        return betaN / betaD

    '''Returns the second derivative of the function'''
    def hessiana(self):
        return np.array([[4, 0],[0, 4]])

    '''Evaluate f at X'''
    def evaluate(self, X):
        return 2*(X[0] - 2)**2 + 2*(X[1] - 3)**2 # Function to be minimized

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

    
    
    
    
