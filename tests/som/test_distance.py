import unittest

import numpy as np
from scipy.spatial import distance

import apollon.som._distance as asd


class TestHellinger(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_unit_distance(self) -> None:
        comp = np.array([[1.0, 0.0, 0.0]])
        sample = np.array([[0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 0.5, 0.5]])
        res = distance.cdist(comp, sample, metric=asd.hellinger)
        self.assertTrue(np.all(res == 1.))


class TestHellinger_stm(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_zero_dist_on_eq_dist(self) -> None:
        n_rows = 5
        sample = np.eye(n_rows).ravel()
        res = asd.hellinger_stm(sample, sample)
        self.assertTrue(np.all(res == 0.0))
