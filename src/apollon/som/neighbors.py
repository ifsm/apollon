# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/som/neighbors.py

Neighborhood computations

Functions:
    gaussian    N-Dimensional Gausian neighborhood.
"""

import numpy as np
from scipy.spatial import distance


def gaussian(grid, center, radius):
    """Compute n-dimensional Gaussian neighbourhood.

    Gaussian neighborhood smoothes the array.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.
    """
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, metric='sqeuclidean')
    return np.exp(-dists/(2*radius**2)).T


def mexican(grid, center, radius):
    """Compute n-dimensional Mexcican hat neighbourhood.

    Mexican hat neighborhood smoothes the array.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.
    """
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, metric='sqeuclidean')
    return ((1-(dists/radius**2)) * np.exp(-dists/(2*radius**2))).T


def star(grid, center, radius):
    """Compute n-dimensional cityblock neighborhood.

    The cityblock neighborhood is a star-shaped area
    around ``center``.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.

    Returns:
    """
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, 'cityblock')
    return (dists <= radius).astype(int).T


def rect(grid, center, radius):
    """Compute n-dimensional Chebychev neighborhood.

    The Chebychev neighborhood is a square-shaped area
    around ``center``.

    Params:
        grid      Array of n-dimensional indices.
        center    Index of the neighborhood center.
        radius    Size of neighborhood.

    Returns:
        Two-dimensional array of in
    """
    center = np.atleast_2d(center)
    dists = distance.cdist(center, grid, 'chebychev')
    return (dists <= radius).astype(int).T
