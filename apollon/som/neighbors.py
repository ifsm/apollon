# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

"""apollon/som/neighbors.py

Neighborhood computations

Functions:
    gaussian    N-Dimensional Gausian neighborhood.
"""

import numpy as np
from scipy.spatial import distance


def gaussian(grid, center, radius):
   """Compute n-dimensional Gaussian neighbourhood on a ``grid``.

   Params:
       grid    Shape of the underlying grid.
       center  Index of of the mode value.
       radius
   """
   center = np.atleast_2d(center)
   dists = distance.cdist(center, grid, metric='sqeuclidean')
   return np.exp(-dists/(2*radius**2)).reshape(-1, 1)

def nh_gaussian_L2(self, center, r):
    """Compute 2D Gaussian neighbourhood around `center`. Distance between
       center and m_i is calculate by Euclidean distance.
    """
    d = _distance.cdist(_np.array(center)[None, :], self._grid,
                       metric='sqeuclidean')
    ssq = 2 * r**2
    return _np.exp(-d/ssq).reshape(-1, 1)
