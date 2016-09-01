import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

'''Gradient descent for 2-dimensions minimization'''
class Gradient():
    
    def __init__(self, alpha = 0.01, precision = 10**-6, X = np.array([[0.] , [0.]])):
         self.Xs = [] # Candidates
         self.F = [] # Evaluations
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
            X = self.Xs[-1] + self.alpha*d
            self.Xs.append(X)
            d = - self.gradient(self.Xs[-1])

        # PLOTTING
        x1s = []
        x2s = []
        for X in self.Xs:
            x1s.append(X[0][0])
            x2s.append(X[1][0])
        print('mock')
        print('x1 =', x1s)
        print('x2 =', x2s)

        plt.plot(x1s, x2s, 'ro')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        
        return self.Xs[-1]

        
         
