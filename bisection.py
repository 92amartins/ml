import numpy as np
import math

def getAlpha(X, d):
    # Initialize lower and upper bounds
    al = 0
    au = 1
    a = (al + au) / 2
    
    while True:
        if(h(a, X, d) > 0):
            au = a
        elif(h(a, X, d) < 0):
            al = a
        else:
            return a
        a = (al + au) / 2

def h(a, X, d):
    gradfT = gradF(X + a*d).T
    return np.dot(gradfT, d)
        

def gradF(X):
    x1 = X[0][0]
    x2 = X[1][0]
        
    return np.array([[4*(x1 - 2)],
                    [4*(x2 - 3)]])
        
        
        
