# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

"""apollon/som/topologies.py

(c) Michael Blaß 2016

Topologies for self-organizing maps.

Functions:
    rect_neighbourhood    Return rectangular neighbourhood.
    vn_neighbourhood      Return 4-neighbourhood.
"""


import numpy as _np


def rect_neighbourhood(mat_shape, point, w=1):
    if point[0] - w < 0:
        rows1 = 0
    else:
        rows1 = point[0] - w
    rows2 = point[0] + w + 1

    if point[1] - w < 0:
        cols1 = 0
    else:
        cols1 = point[1] - w
    cols2 = point[1] + w + 1

    mask = _np.ones(mat_shape)
    mask[rows1:rows2, cols1:cols2] = 0
    mask[point] = 1
    out = _np.ma.masked_array(mask, mask=mask)
    return out


def vn_neighbourhood(x, y, dx, dy, flat=False):
    """ Compute Von Neuman Neighbourhood.

    Compute the Von Neuman Neighbourhood of index (x, y) given an array with
    dimension (dx, dy). The Von Neumann Neighbourhood is the 4-neighbourhood,
    which includes the four direct neighbours of index (x, y) given a rect-
    angular array.

    Params:
        x    (int)       x-Index for which to compute the neighbourhood.
        y    (int)       y-Index for which to compute the neighbourhood.
        dx   (int)       Size of enclosing array's x-axis.
        dy   (int)       Size of enclosing array's y-axis.
        flat (bool)      Return flat index if True. Default is False.

    Return:
        1d-array of ints if flat, 2d-array otherwise.
    """
    nh = []

    if x-1 >= 0:
        nh.append((x-1, y))
    if x+1 < dx:
        nh.append((x+1, y))
    if y-1 >= 0:
        nh.append((x, y-1))
    if y+1 < dy:
        nh.append((x, y+1))

    if flat:
        nh = _np.array(nh)
        return _np.ravel_multi_index(nh.T, (dx, dy))
    else:
        return nh
