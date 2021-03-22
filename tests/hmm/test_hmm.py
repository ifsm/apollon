"""hmm_test.py
Unit test for HMM implementation."""


import numpy as np
from scipy.stats import poisson
import unittest

from apollon.hmm.poisson import PoissonHmm


class TestHMM_utilities(unittest.TestCase):
    def setUp(self):
        # Arbitrary transition probability matrix
        self.A = np.array([[1., 0, 0], [.2, .3, .5], [.1, .3, .6]])

        # Wrong number of dimensions
        self.B1 = np.array([1., 0, 0, 0])
        self.B2 = np.array([[[1., 0, 0], [.2, .3, .5], [.1, .3, .6]]])

        # Not quadratic
        self.C1 = np.array([[1., 0, 0], [.2, .3, .5]])
        self.C2 = np.array([[1.0], [.5, .5], [.2, .8]])

        # Rows do not sum up to one
        self.D = np.array([[.2, .3, .5], [.5, .4, .2], [1., 0, 0]])

    def test_success(self):
        mus = [20, 40, 80, 120, 40]
        m = len(mus)
        data = np.concatenate([poisson(mu).rvs(30) for mu in mus])
        hmm = PoissonHmm(data, m)
        hmm.fit(data)
        self.assertTrue(hmm.success)


if __name__ == '__main__':
    unittest.main()
