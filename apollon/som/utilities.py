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


def decrease_linear(start, step, stop=1):
    '''Linearily decrease `start`  in `step` steps to `stop`.'''
    a = (stop - start) / (step-1)
    for x in range(step):
        yield a * x + start


def decrease_expo(start, step, stop=1):
    '''Exponentially decrease `start`  in `step` steps to `stop`.'''
    b = _np.log(stop / start) / (step-1)
    for x in range(step):
        yield start * _np.exp(b*x)


def umatrix(lattice, dx, dy, metric='euclidean', w=1, normed=True):
    dxy = dx, dy
    out = zeros(dxy)

    for i in ndindex(dxy):
        nb = _rect_neighbourhood(dxy, i, w=1)
        i_flat = ravel_multi_index(i, dxy)
        out[i] = distance.cdist(lattice[i_flat, None],
                                lattice[~nb.mask.flatten()],
                                metric=metric, p=2).sum()
    if norm:
        return out / _np.max(out)
    else:
        return out
