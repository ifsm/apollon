import unittest
import numpy as np

from apollon import fractal
from apollon.signal.tools import sinusoid, noise


class Test_ModuleFractal(unittest.TestCase):
    def setUp(self):
        self.data = sinusoid(300, fs=3000) + noise(.1, 3000)

    def test_CorrelationDimension(self):
        cdim = fractal.correlation_dimension(self.data, 12, 80, 100)
        self.assertTrue(isinstance(cdim, float))

if __name__ == '__main__':
    unittest.main()
