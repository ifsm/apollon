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

from apollon.som.topologies import _rect_neighbourhood

def activation_map(som, **kwargs):
    ax = _plt.gca()
    am = som.activation_map.reshape(som.shape[:2])
    ax.imshow(_np.flipud(am), vmin=0, vmax=som.activation_map.max(), **kwargs)

def distance_map(som, w=1, ax=None, plotit=True, **kwargs):
    dx, dy = som.shape[:2]
    out = _np.zeros(dx*dy)
    X, Y = _np.meshgrid(range(dx), range(dy), indexing='ij')
    for i in range(len(som.lattice)):
        n = _np.unravel_index(i, (dx, dy))
        M = _rect_neighbourhood((dx, dy), n, w)
        flat_idx = [i*dx+j for i, j in zip(X[~M.mask], Y[~M.mask])]

        dd = _distance.cdist(som.lattice[i, _np.newaxis],
                            som.lattice[flat_idx])
        out[i] = dd.sum() / len(dd)

    out /= _np.abs(out).max()   # caution! dtype must be float!
    out = out.reshape(dx, dy)

    if plotit:
        if ax is None:
            ax = _plt.gca()
        ax.imshow(out, vmin=0, vmax=1, **kwargs)
    return out


def distance_map3d(som, w=1, ax=None, plotit=True, **kwargs):
    dx, dy = som.shape[:2]
    out = _np.zeros(dx*dy)
    X, Y = _np.meshgrid(range(dx), range(dy), indexing='ij')
    for i in range(len(som.lattice)):
        n = _np.unravel_index(i, (dx, dy))
        M = _rect_neighbourhood((dx, dy), n, w)
        flat_idx = [i*dx+j for i, j in zip(X[~M.mask], Y[~M.mask])]

        dd = _distance.cdist(som.lattice[i, _np.newaxis],
                            som.lattice[flat_idx])
        out[i] = dd.sum() / len(dd)

    out /= _np.abs(out).max()   # caution! dtype must be float!
    out = out.reshape(dx, dy)

    if plotit:
        fig = _plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, _np.flipud(out), vmin=0, vmax=1,
                        cstride=1, rstride=1, **kwargs)
    return out
