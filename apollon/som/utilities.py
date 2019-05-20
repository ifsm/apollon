# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""apollon/som/uttilites.py

(c) Michael Blaß, 2016

Utilites for self.organizing maps.

Functions:
    activation_map    Plot activation map
    distance_map      Plot a distance map
    distance_map3d    Plot a 3d distance map
"""

import matplotlib.pyplot as _plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as _np
from scipy.spatial import distance as _distance

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
