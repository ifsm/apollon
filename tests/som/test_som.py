import unittest
from typing import Tuple

from hypothesis import given
import hypothesis.strategies as hst
import numpy as np
import scipy as sp

from apollon.som.som import SomBase, SomGrid

SomDim = Tuple[int, int, int]
dimension = hst.integers(min_value=1, max_value=100)
som_dims = hst.tuples(dimension, dimension, dimension)


class TestSomBase(unittest.TestCase):
    @given(som_dims)
    def test_dims(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertEqual(som.dims, dims)

    @given(som_dims)
    def test_dx(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertEqual(som.dx, dims[0])

    @given(som_dims)
    def test_dy(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertEqual(som.dy, dims[1])

    @given(som_dims)
    def test_dw(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertEqual(som.dw, dims[2])

    @given(som_dims)
    def test_n_units(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertEqual(som.n_units, dims[0]*dims[1])

    @given(som_dims)
    def test_shape(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertEqual(som.shape, (dims[0], dims[1]))

    @given(som_dims)
    def test_grid(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertIsInstance(som.grid, SomGrid)

    """
    @given(som_dims)
    def test_dists(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertIsInstance(som.dists, np.ndarray)
    """

    @given(som_dims)
    def test_weights(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        self.assertIsInstance(som.weights, np.ndarray)

    @given(som_dims)
    def test_match(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'uniform', 'euclidean')
        data = np.random.rand(100, dims[2])
        self.assertIsInstance(som.match(data), np.ndarray)


"""
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
"""

if __name__ == '__main__':
    unittest.main()
