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


def umatrix(lattice, dx, dy, metric='euclidean', w=1, isNormed=True):
    dxy = dx, dy
    out = _np.zeros(dxy)

    for i in _np.ndindex(dxy):
        nb = _topologies.rect_neighbourhood(dxy, i, w=1)
        i_flat = _np.ravel_multi_index(i, dxy)
        out[i] = _distance.cdist(lattice[i_flat, None],
                                lattice[~nb.mask.flatten()],
                                metric=metric, p=2).sum()
    if isNormed:
        return out / _np.max(out)
    else:
        return out
