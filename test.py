import unittest
import numpy as np
import gradient as g
import conjugate as c
import matplotlib.pyplot as plt

class TestGradient(unittest.TestCase):

    def setUp(self):
        self.grad = g.Gradient(0.4)
        self.conj = c.Conjugate(0.4)

    def testGradient(self):
        X = np.array([[0.0],[0.0]])
        x1 = X[0][0]
        x2 = X[1][0]
        expected = np.array([[4*(x1-2)], [4*(x2-3)]])
        result = self.conj.gradient(X)
        self.assertTrue(np.allclose(expected, result))
        
    def testConjugate(self):
        expected = np.array([[2.0], [3.0]])
        result = self.conj.descent()
        self.assertTrue(np.allclose(expected, result))

    def testGradientDescent(self):
        expected = np.array([[2.0], [3.0]])
        result = self.grad.descent()
        self.assertTrue(np.allclose(expected, result))

        
        
if __name__ == '__main__':
    unittest.main()


 
