"""apollon/som/utilites.py

Utilities for self.organizing maps.

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net
"""
import itertools
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
from scipy.spatial import distance as _distance
from scipy import stats as _stats

from apollon.types import Array, Shape
from apollon import tools


def grid_iter(n_rows: int, n_cols: int) -> Iterator[Tuple[int, int]]:
    """Compute grid indices of an two-dimensional array.

    Args:
        n_rows:  Number of array rows.
        n_cols:  Number of array columns.

    Returns:
        Multi-index iterator.
    """
    return itertools.product(range(n_rows), range(n_cols))


def grid(n_rows: int, n_cols: int) -> Array:
    """Compute grid indices of a two-dimensional array.

    Args:
        n_rows:  Number of array rows.
        n_cols:  Number of array columns.

    Returns:
        Two-dimensional array in which each row represents an multi-index.
    """
    return np.array(list(grid_iter(n_rows, n_cols)))


def decrease_linear(start: float, step: float, stop: float = 1.0
                    ) -> Iterator[float]:
    """Linearily decrease ``start``  in ``step`` steps to ``stop``."""
    if step < 1 or not isinstance(step, int):
        raise ValueError('Param `step` must be int >= 1.')
    elif step == 1:
        yield start
    else:
        a = (stop - start) / (step-1)
        for x in range(step):
            yield a * x + start


def decrease_expo(start: float, step: float, stop: float = 1.0
                  ) -> Iterator[float]:
    """Exponentially decrease ``start``  in ``step`` steps to ``stop``."""
    if step < 1 or not isinstance(step, int):
        raise ValueError('Param `step` must be int >= 1.')
    elif step == 1:
        yield start
    else:
        b = np.log(stop / start) / (step-1)
        for x in range(step):
            yield start * np.exp(b*x)

"""
def match(weights: Array, data: Array, kth, metric: str):
    dists = _distance.cdist(weights, data, metric)
    idx = dists.argpartition(kth, axis=0)
    min_vals = dists[min_idx]
    return (min_idx, min_vals)
"""

def best_match(weights: Array, inp: Array, metric: str):
    """Compute the best matching unit of ``weights`` for each
    element in ``inp``.

    If several elemets in ``weights`` have the same distance to the
    current element of ``inp``, the first element of ``weights`` is
    choosen to be the best matching unit.

    Args:
        weights:    Two-dimensional array of weights, in which each row
                    represents an unit.
        inp:        Array of test vectors. If two-dimensional, rows are
                    assumed to represent observations.
        metric:     Distance metric to use.

    Returns:
        Index and error of best matching units.
    """
    if weights.ndim != 2:
        msg = (f'Array ``weights`` has {weights.ndim} dimensions, it '
               'has to have exactly two dimensions.')
        raise ValueError(msg)

    if weights.shape[-1] != inp.shape[-1]:
        msg = (f'Feature dimension of ``weights`` has {weights.shape[0]} '
               'elemets, whereas ``inp`` has {inp.shape[-1]} elemets. '
               'However, both dimensions have to match exactly.')
        raise ValueError(msg)

    inp = np.atleast_2d(inp)
    if inp.ndim > 2:
        msg = (f'Array ``inp`` has {weights.ndim} dimensions, it '
               'has to have one or two dimensions.')
        raise ValueError(msg)

    dists = _distance.cdist(weights, inp, metric)
    return dists.argmin(axis=0), dists.min(axis=0)


def init_pca(data: Array, shape: Shape, adapt: bool = True) -> Array:
    """Compute initial SOM weights by the first two principal components of the
    input data.

    Args:
        data:   Input data set.
        shape:  Shape of SOM.
        adapt:  If ``True``, the largest value of ``shape`` is applied to the
                principal component with the largest sigular value. This
                orients the map, such that map dimension with the most units
                coincides with principal component with the largest variance.

    Returns:
        Matrix of SOM weights.
    """
    vals, vects, trans_data = tools.pca(data, 2)
    data_limits = np.column_stack((trans_data.min(axis=0),
                                   trans_data.max(axis=0)))
    if adapt:
        shape = sorted(shape, reverse=True)
    dim_x = np.linspace(*data_limits[0], shape[0])
    dim_y = np.linspace(*data_limits[1], shape[1])
    grid_x, grid_y = np.meshgrid(dim_x, dim_y)
    points = np.vstack((grid_x.ravel(), grid_y.ravel()))
    weights = points.T @ vects + data.mean(axis=0)
    return weights


def init_simplex(n_features, n_units):
    """Initialize the weights with stochastic matrices.

    The rows of each n by n stochastic matrix are sampes drawn from the
    Dirichlet distribution, where n is the number of rows and cols of the
    matrix. The diagonal elemets of the matrices are set to twice the
    probability of the remaining elements.
    The square root n of the weight vectors' size must be element of the
    natural numbers, so that the weight vector is reshapeable to a square
    matrix.

    Args:
        n_features:  Number of features in each vector.
        n_units:     Number of units on the SOM.

    Returns:
        Two-dimensional array of shape (n_units, n_features), in which each
        row is a flattened random stochastic matrix.
    """
    # check for square matrix
    n_rows = np.sqrt(n_features)
    if bool(n_rows - int(n_rows)):
        msg = (f'Weight vector (len={n_features}) is not '
               'reshapeable to square matrix.')
        raise ValueError(msg)
    else:
        n_rows = int(n_rows)

    # set alpha
    alpha = np.full((n_rows, n_rows), 500)
    np.fill_diagonal(alpha, 1000)

    # sample from dirichlet distributions
    st_matrix = np.hstack([_stats.dirichlet.rvs(alpha=a, size=n_units)
                           for a in alpha])
    return st_matrix


def distribute(bmu_idx: Iterable[int], n_units: int
               ) -> Dict[int, List[int]]:
    """List training data matches per SOM unit.

    This method assumes that the ith element of ``bmu_idx`` corresponds to the
    ith vetor in a array of input data vectors.

    Empty units result in empty list.

    Args:
        bmu_idx:  Indices of best matching units.
        n_units:  Number of units on the SOM.

    Returns:
        Dictionary in which the keys represent the flat indices of SOM units.
        The corresponding value is a list of indices of those training data
        vectors that have been mapped to this unit.
    """
    unit_matches = {i:[] for i in range(n_units)}
    for data_idx, bmu in enumerate(bmu_idx):
        unit_matches[bmu].append(data_idx)
    return unit_matches
