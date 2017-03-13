#!python3
# -*- coding: utf-8 -*-


"""apollon/som/topologies.py

(c) Michael Blaß 2016

Topologies for self-organizing maps.

Functions:
    _rect_neighbourhood    Return rectangular neighbourhood.
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
