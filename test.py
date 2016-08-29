import unittest
import gradient as g
import numpy as np

class TestGradient(unittest.TestCase):

    def setUp(self):
        self.grad = g.Gradient2D(0.15)

    def testDerivative(self):
        for x in range(10000):
            self.assertEqual(self.grad.derivative('x1', x), 4*(x-2))
            self.assertEqual(self.grad.derivative('x2', x), 4*(x-3))

    def testD(self):
        self.assertEqual(list(self.grad.d([0., 0.])), [8. , 12.])

    def testDescent(self):
        self.assertEqual(list(self.grad.descent()), [2., 3.])
        
        
if __name__ == '__main__':
    unittest.main()
