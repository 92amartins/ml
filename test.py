import unittest
import numpy as np
import gradient as g
import conjugate as c
import matplotlib.pyplot as plt

class TestGradient(unittest.TestCase):

    def setUp(self):
        self.grad = g.Gradient(0.01)
        self.conj = c.Conjugate()

    def testGradientDescent(self):
        expected = np.array([[2.0], [3.0]])
        result = self.grad.descent()
        self.grad.plot('Gradiente Descendente', 'X1', 'X2')
        self.assertTrue(np.allclose(expected, result))

    def testConjugateGradient(self):
        expected = np.array([[2.0], [3.0]])
        result = self.conj.descent()
        self.conj.plot('Gradiente Conjugado', 'X1', 'X2')
        self.assertTrue(np.allclose(expected, result))


    def testGradient(self):
        X = np.array([[0.0],[0.0]])
        x1 = X[0][0]
        x2 = X[1][0]
        expected = np.array([[4*(x1-2)], [4*(x2-3)]])
        result = self.conj.gradient(X)
        self.assertTrue(np.allclose(expected, result))

        X = np.array([[0.0],[0.0]])
        x1 = X[0][0]
        x2 = X[1][0]
        expected = np.array([[4*(x1-2)], [4*(x2-3)]])
        result = self.grad.gradient(X)
        self.assertTrue(np.allclose(expected, result))

    def testHessiana(self):
        expected = np.array([[4, 0], [0, 4]])
        result = self.conj.hessiana()
        self.assertTrue(np.allclose(expected, result))

    def testAlpha(self):
        X = np.array([[0.0],[0.0]])
        d = - self.conj.gradient(X)
        H = self.conj.hessiana()

        expected = 0.25
        result = self.conj.alpha(d, H)
        self.assertEqual(expected, result)

    
    def testBeta(self):
        expected = -1
        X = np.array([[0.0],[0.0]])
        d = - self.conj.gradient(X)
        gT = (self.conj.gradient(X)).T
        H = self.conj.hessiana()
        
        result = self.conj.beta(gT, H, d)

        self.assertEqual(expected, result)

    

    

    
        
        
if __name__ == '__main__':
    unittest.main()


 
