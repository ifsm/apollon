# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/som/neighbors.py

Neighborhood computations

Functions:
    gaussian    N-Dimensional Gausian neighborhood.
"""
from typing import List, Tuple

import numpy as np
from scipy.spatial import distance

from .. types import Array

Shape = Tuple[int, int]
Coord = Tuple[int, int]
AdIndex = Tuple[List[int], List[int]]


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


def neighborhood(grid, metric='sqeuclidean'):
    """Compute n-dimensional cityblock neighborhood.

    The cityblock neighborhood is a star-shaped area
    around ``center``.

    Params:
        grid:      Array of n-dimensional indices.
        metric:    Distance metric.

    Returns:
        Pairwise distances of map units.
    """
    return distance.squareform(distance.pdist(grid, metric))


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

""" NNNNNNNNNEEEEEEEEEEWWWWWW STUFFFFFFFF """
def gauss_kern(nhb, r):
    return np.exp(-nhb/(r))


def is_neighbour(cra: Array, crb: Array, grid: Array, metric: str) -> Array:
    """Compute neighbourship between each coordinate in ``units_a`` abd
    ``units_b`` on ``grid``.

    Args:
        cra:     (n x 2) array of grid coordinates.
        crb:     (n x 2) array of grid coordinates.
        grid:    SOM grid array.
        metric:  Name of distance metric function.

    Returns:
        One-dimensional boolean array. ``True`` in position n means that the
        points ``cra[n]`` and ``crb[n]`` are direct neighbours on ``grid``
        regarding ``metric``.
    """


def check_bounds(shape: Shape, point: Coord) -> bool:
    """Return ``True`` if ``point`` is valid index in ``shape``.

    Args:
        shape:  Shape of two-dimensional array.
        point:  Two-dimensional coordinate.

    Return:
        True if ``point`` is within ``shape`` else ``False``.
    """
    return (0 <= point[0] < shape[0]) and (0 <= point[1] < shape[1])


def direct_rect_nb(shape: Shape, point: Coord) -> AdIndex:
    """Return the set of direct neighbours of ``point`` given rectangular
    topology.

    Args:
        shape:  Shape of two-dimensional array.
        point:  Two-dimensional coordinate.

    Returns:
        Advanced index of points in neighbourhood set.
    """
    nhb = []
    for i in range(point[0]-1, point[0]+2):
        for j in range(point[1]-1, point[1]+2):
            if check_bounds(shape, (i, j)):
                nhb.append((i, j))
    return np.asarray(nhb)

