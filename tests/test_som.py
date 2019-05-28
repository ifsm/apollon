#!/usr/bin/python3


import unittest
import numpy as np
import scipy as sp

from apollon.som.som import SelfOrganizingMap


class TestSelfOrganizingMap(unittest.TestCase):
    def setUp(self):
        N = 100

        m1 = (0, 0)
        m2 = (10, 15)
        c1 = ((10, 0), (0, 10))
        c2 = ((2, 0), (0, 2))

        seg1 = np.random.multivariate_normal(m1, c1, N)
        seg2 = np.random.multivariate_normal(m2, c2, N)

        self.data = np.vstack((seg1, seg2))
        self.dims = (10, 10, 2)

    def test_init_random(self):
        som = SelfOrganizingMap(self.dims, init_distr='uniform')
        self.assertTrue('weights', True)

if __name__ == '__main__':
    unittest.main()
