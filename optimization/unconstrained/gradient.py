import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import bisection as bis

'''Gradient descent for 2-dimensions minimization'''
class Gradient():
    
    def __init__(self, alpha = 0.01, precision = 10**-6, X = np.array([[0.] , [0.]])):
         self.Xs = [] # Candidates
         self.Xs.append(X) # Append first candidate (X0)
         self.alpha = alpha   # Learning rate
         self.precision = precision # Stop criteria

    '''Returns the gradient of f(x1, x2) evaluated at X'''
    def gradient(self, X):
        x1 = X[0][0]
        x2 = X[1][0]
        
        return np.array([[4*(x1 - 2)],
                         [4*(x2 - 3)]])

    '''Evaluate f at X'''
    def evaluate(self, X):
        return 2*(X[0] - 2)**2 + 2*(X[1] - 3)**2 # Function to be minimized

    '''Returns the point (x1, x2) which minimizes f'''
    def descent(self):
        d = - self.gradient(self.Xs[0])

        # Calculate new X and Update direction
        while la.norm(d) > self.precision:
            ialpha = bis.getAlpha(self.Xs[-1], d) # Iteration alpha from bisection
            X = self.Xs[-1] + ialpha*d
            self.Xs.append(X)
            d = - self.gradient(self.Xs[-1])
        return self.Xs[-1]

    def plot(self, title, xlabel, ylabel):
        x1s = []
        x2s = []
        
        for X in self.Xs:
            x1s.append(X[0][0])
            x2s.append(X[1][0])

        plt.plot(x1s, x2s, 'bo')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        

        
         
