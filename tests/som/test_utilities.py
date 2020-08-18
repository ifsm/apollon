import unittest

from hypothesis import strategies as hst
from hypothesis import given
import numpy as np
from scipy.spatial import distance

from apollon.som import utilities as asu
from apollon.types import SomDims

dimension = hst.integers(min_value=2, max_value=50)
som_dims = hst.tuples(dimension, dimension, dimension)
"""
class TestMatch(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = np.random.rand(100, 5)
        self.data = np.random.rand(200, 5)

    def test_returns_tuple(self) -> None:
        res = asu.match(self.weights, self.data, 2, 'euclidean')
        self.assertIsInstance(res, tuple)

    def test_elements_are_arrays(self) -> None:
        bmu, err = asu.match(self.weights, self.data, 'euclidean')
        self.assertIsInstance(bmu, np.ndarray)
        self.assertIsInstance(err, np.ndarray)

    def test_correct_ordering(self) -> None:
        kth = 5
        bmu, err = asu.match(self.weights, self.data, 'euclidean')
        wdists = distance.cdist(self.weights, self.data)
        kswd = wdists.sort(axis=0)[:kth, :]
"""

class TestDistribute(unittest.TestCase):
    def setUp(self) -> None:
        self.n_units = 400
        self.bmu = np.random.randint(0, self.n_units, 100)

    def returns_dict(self):
        res = asu.distribute(self.bmu, self.n_units)
        self.assertIsInstance(res, dict)


class TestSampleHist(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @given(som_dims)
    def test_rows_are_stochastic(self, dims: SomDims) -> None:
        weights = asu.sample_hist(dims)
        comp =np.isclose(weights.sum(axis=1), 1)
        self.assertTrue(comp.all())


class TestSamplePca(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @given(som_dims)
    def test_x(self, dims: SomDims) -> None:
        weights = asu.sample_pca(dims)
"""
class TestSelfOrganizingMap(unittest.TestCase):
    def setUp(self):
        self.weights = np.load('data/test_weights.npy')
        self.inp = np.load('data/test_inp.npy')

    def test_best_match_computation(self):
        test_bmu = np.load('data/bmu_idx_euc.npy')
        test_err = np.load('data/bmu_err_euc.npy')
        bmu, err = utilities.best_match(self.weights, self.inp, 'euclidean')
        self.assertTrue(np.array_equiv(test_bmu, bmu))
        self.assertTrue(np.array_equiv(test_err, err))
"""

if __name__ == '__main__':
    unittest.main()
