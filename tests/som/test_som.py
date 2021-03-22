import unittest
from typing import Tuple

from hypothesis import given
import hypothesis.strategies as hst
import numpy as np
import scipy as sp

from apollon.som.som import SomBase, SomGrid

SomDim = Tuple[int, int, int]
dimension = hst.integers(min_value=2, max_value=50)
som_dims = hst.tuples(dimension, dimension, dimension)


class TestSomBase(unittest.TestCase):

    @given(som_dims)
    def test_dims(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertEqual(som.dims, dims)

    @given(som_dims)
    def test_dx(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertEqual(som.dx, dims[0])

    @given(som_dims)
    def test_dy(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertEqual(som.dy, dims[1])

    @given(som_dims)
    def test_dw(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertEqual(som.dw, dims[2])

    @given(som_dims)
    def test_n_units(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertEqual(som.n_units, dims[0]*dims[1])

    @given(som_dims)
    def test_shape(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertEqual(som.shape, (dims[0], dims[1]))

    @given(som_dims)
    def test_grid(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertIsInstance(som.grid, SomGrid)

    """
    @given(som_dims)
    def test_dists(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertIsInstance(som.dists, np.ndarray)
    """

    @given(som_dims)
    def test_weights(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        self.assertIsNone(som.weights)

    @given(som_dims)
    def test_match(self, dims: SomDim) -> None:
        data = np.random.rand(100, dims[2])
        som = SomBase(dims, 10, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        som._weights = som.init_weights(data, som.shape)
        self.assertIsInstance(som.match(data), np.ndarray)

    @given(som_dims)
    def test_umatrix_has_map_shape(self, dims: SomDim) -> None:
        data = np.random.rand(100, dims[2])
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        som._weights = som.init_weights(data, som.shape)
        um = som.umatrix()
        self.assertEqual(um.shape, som.shape)

    @given(som_dims)
    def test_umatrix_scale(self, dims: SomDim) -> None:
        som = SomBase(dims, 100, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        som._weights = np.tile(np.arange(som.n_features), (som.n_units, 1))
        som._weights[:, -1] = np.arange(som.n_units)
        um = som.umatrix(scale=True, norm=False)
        self.assertEqual(um[0, 0], um[-1, -1])
        self.assertEqual(um[0, -1], um[-1, 0])

    @given(som_dims)
    def test_umatrix_norm(self, dims: SomDim) -> None:
        data = np.random.rand(100, dims[2])
        som = SomBase(dims, 10, 0.1, 10, 'gaussian', 'rnd', 'euclidean')
        som._weights = som.init_weights(data, som.shape)
        um = som.umatrix(norm=True)
        self.assertEqual(um.max(), 1.0)


if __name__ == '__main__':
    unittest.main()
