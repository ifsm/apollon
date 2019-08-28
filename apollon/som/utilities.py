# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

"""apollon/som/uttilites.py

Utilities for self.organizing maps.

Functions:
    activation_map    Plot activation map
    distance_map      Plot a distance map
    distance_map3d    Plot a 3d distance map
"""

import matplotlib.pyplot as _plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as _np
from scipy.spatial import distance as _distance
from scipy import stats as _stats
import apollon.som.topologies as _topologies


def activation_map(som, **kwargs):
    ax = _plt.gca()
    am = som.activation_map.reshape(som.shape[:2])
    ax.imshow(_np.flipud(am), vmin=0, vmax=som.activation_map.max(), **kwargs)


def decrease_linear(start, step, stop=1):
    '''Linearily decrease `start`  in `step` steps to `stop`.'''
    if step < 1 or not isinstance(step, int):
        raise ValueError('Param `step` must be int >= 1.')
    elif step == 1:
        yield start
    else:
        a = (stop - start) / (step-1)
        for x in range(step):
            yield a * x + start


def decrease_expo(start, step, stop=1):
    '''Exponentially decrease `start`  in `step` steps to `stop`.'''
    if step < 1 or not isinstance(step, int):
        raise ValueError('Param `step` must be int >= 1.')
    elif step == 1:
        yield start
    else:
        b = _np.log(stop / start) / (step-1)
        for x in range(step):
            yield start * _np.exp(b*x)


def get_winner(weights, test_vector, metric='euclidean'):
    if test_vector.ndim == 1:
        d = _distance.cdist(test_vector[None, :], weights, metric=metric)
        return _np.argmin(d)
    elif test_vector.ndim == 2:
        ds = _distance.cdist(test_vector, weights, metric=metric)
        return _np.argmin(ds, axis=1)
    else:
        raise ValueError('Wrong dimension of input data: {}'.format(data.ndim))


def umatrix(weights, dxy, metric='euclidean'):
    """ Compute unified distance matrix.

    Params:
        weights (ndarray)    SOM weights matrix.
        dxy (tuple)
        metric (str)         Metric to use.

    Return:
        (ndarray)    unified distance matrix.
    """
    out = _np.empty(dxy, dtype='float64')

    for i, mi in enumerate(_np.ndindex(dxy)):
        nh_flat_idx = _topologies.vn_neighbourhood(*mi, *dxy, flat=True)
        p = weights[i][None]
        nh = weights[nh_flat_idx]
        out[mi] = _distance.cdist(p, nh).sum() / len(nh)

    return out / out.max()


def init_simplex(n_features, n_units):
    """Initialize the weights with stochastic matrices.

    The rows of each n by n stochastic matrix are sampes drawn from the
    Dirichlet distribution, where n is the number of rows and cols of the
    matrix. The diagonal elemets of the matrices are set to twice the
    probability of the remaining elements.
    The square root n of the weight vectors' size must be element of the
    natural numbers, so that the weight vector is reshapeable to a square
    matrix.

    Params:
        n_features    Number of features in each vector.
        n_units       Number of units on the SOM.

    Returns:
        Two-dimensional array of shape (n_units, n_features), in which each
        row is a flattened random stochastic matrix.
    """
    # check for square matrix
    n_rows = _np.sqrt(n_features)
    if bool(n_rows - int(n_rows)):
        raise ValueError(f'Weight vector (len={n_features}) is not'
                'reshapeable to square matrix.')
    else:
        n_rows = int(n_rows)

    # set alpha
    alpha = _np.full((n_rows, n_rows), 500)
    _np.fill_diagonal(alpha, 1000)

    # sample from dirichlet distributions
    st_matrix = _np.hstack([_stats.dirichlet.rvs(alpha=a, size=n_units)
                            for a in alpha])
    return st_matrix
