import unittest
import gradient as g
import numpy as np

class TestGradient(unittest.TestCase):

    def setUp(self):
        self.grad = g.Gradient(0.25)

    def testDescent(self):
        self.assertEqual(list(self.grad.descent()), [2.0, 3.0])
        
if __name__ == '__main__':
    unittest.main()
