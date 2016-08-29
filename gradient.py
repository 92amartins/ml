import numpy as np
from numpy import linalg as la
class Gradient2D():
    
    def __init__(self, alpha = 0.01, precision = 10**-6, X = [0. , 0.]):
         self.Xs = [] # Candidates
         self.F = [] # Evaluations
         self.Xs.append(X) # Append first candidate (X0)
         self.alpha = alpha   # Learning rate
         self.precision = precision # Stop criteria

    '''Returns the evaluation of the partial derivative'''
    def derivative(self, dx, x):
        if(dx == 'x1'):
            return 4*(x-2)
        elif(dx == 'x2'):
            return 4*(x-3)

    '''Calculates the vector d'''
    def d(self, X):
        D = np.array([-self.derivative('x1', X[0]),
             -self.derivative('x2', X[1])])
        return D

    '''Evaluate f at X'''
    def f(self, X):
        return 2*(X[0] - 2)**2 + 2*(X[1] - 3)**2 # Function to be minimized

    '''Returns the point (x1, x2) which provides the minimum evaluation to
       f'''
    def descent(self):
        while la.norm(self.d(self.Xs[-1])) >= self.precision:
            self.F.append(self.f(self.Xs[-1] + self.alpha*self.d(self.Xs[-1])))
            self.Xs.append(self.Xs[-1] + self.alpha*self.d(self.Xs[-1]))
        return self.Xs[-1]
    
    
