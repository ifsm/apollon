import unittest

import numpy as np
from scipy.stats import multivariate_normal

from apollon import tools


class TestPca(unittest.TestCase):
    def setUp(self) -> None:
        mu = (0, 0)
        cov = ((10, 0), (0, 12))
        n = 1000
        self.data = multivariate_normal(mu, cov).rvs(n)

    def test_output_is_tuple(self):
        self.assertIsInstance(tools.pca(self.data, 2), tuple)

if __name__ == '__main__':
    unittest.main()
