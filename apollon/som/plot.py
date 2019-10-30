# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/som/plot.py
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from apollon import tools
from apollon.types import Array


def plot_calibration(self, lables=None, ax=None, cmap='plasma', **kwargs):
    """Plot calibrated map.

    Args:
        labels:
        ax
        cmap:

    Returns:
    """
    if not self.isCalibrated:
        raise ValueError('Map not calibrated.')
    else:
        if ax is None:
            fig, ax = _new_axis()
        ax.set_title('Calibration')
        ax.set_xlabel('# units')
        ax.set_ylabel('# units')
        ax.imshow(self._cmap.reshape(self.dx, self.dy), origin='lower',
                  cmap=cmap)
        #return ax


def plot_datamap(self, data, targets, interp='None', marker=False,
                 cmap='viridis', **kwargs):
    """Represent the input data on the map by retrieving the best
    matching unit for every element in `data`. Mark each map unit
    with the corresponding target value.

    Args:
        data:    Input data set.
        targets: Class labels or values.
        interp:  matplotlib interpolation method name.
        marker:  Plot markers in bmu position if True.

    Returns:
       axis, umatrix, bmu_xy
    """
    ax, udm = self.plot_umatrix(interp=interp, cmap=cmap, **kwargs)

    #
    # TODO: Use .transform() instead
    #
    bmu, err = self.get_winners(data)

    x, y = _np.unravel_index(bmu, (self.dx, self.dy))
    fd = {'color':'#cccccc'}
    if marker:
        ax.scatter(y, x, s=40, marker='x', color='r')

    for i, j, t in zip(x, y, targets):
        ax.text(j, i, t, fontdict=fd,
                horizontalalignment='center',
                verticalalignment='center')
    return (ax, udm, (x, y))


def plot_qerror(self, ax=None, **kwargs):
    """Plot quantization error."""
    if ax is None:
        fig, ax = _new_axis(**kwargs)

    ax.set_title('Quantization Errors per iteration')
    ax.set_xlabel('# interation')
    ax.set_ylabel('Error')

    ax.plot(self.quantization_error, lw=3, alpha=.8,
            label='Quantizationerror')


def plot_umatrix(self, interp='None', cmap='viridis', ax=None, **kwargs):
    """Plot unified distance matrix.

    The unified distance matrix (udm) allows to visualize weight matrices
    of high dimensional weight vectors. The entries (x, y) of the udm
    correspondto the arithmetic mean of the distances between weight
    vector (x, y) and its 4-neighbourhood.

    Args:
        w:        Neighbourhood width.
        interp:   matplotlib interpolation method name.
        ax:       Provide custom axis object.

   Returns:
       axis, umatrix
    """
    if ax is None:
        fig, ax = aplot._new_axis()
    udm = _som_utils.umatrix(self.weights, self.shape, metric=self.metric)

    ax.set_title('Unified distance matrix')
    ax.set_xlabel('# units')
    ax.set_ylabel('# units')
    ax.imshow(udm, interpolation=interp, cmap=cmap, origin='lower')
    return ax, udm


def plot_umatrix3d(self, w=1, cmap='viridis', **kwargs):
    """Plot the umatrix in 3d. The color on each unit (x, y) represents its
       mean distance to all direct neighbours.

    Args:
        w: Neighbourhood width.

    Returns:
        axis, umatrix
    """
    fig, ax = _new_axis_3d(**kwargs)
    udm = _som_utils.umatrix(self.weights, self.shape, metric=self.metric)
    X, Y = _np.mgrid[:self.dx, :self.dy]
    ax.plot_surface(X, Y, udm, cmap=cmap)
    return ax, udm


def plot_features(self, figsize=(8, 8)):
    """Values of each feature of the weight matrix per map unit.

    This works currently ony for feature vectors of len dw**2.

    Args:
        Size of figure.
    """
    d = _np.sqrt(self.dw).astype(int)
    rweigths = self.weights.reshape(self.dims)

    fig, _ = _plt.subplots(d, d, figsize=figsize, sharex=True, sharey=True)
    for i, ax in enumerate(fig.axes):
        ax.axison=False
        ax.imshow(rweigths[..., i], origin='lower')


def plot_whist(self, interp='None', ax=None, **kwargs):
    """Plot the winner histogram.

    The darker the color on position (x, y) the more often neuron (x, y)
    was choosen as winner. The number of winners at edge neuros is
    magnitudes of order higher than on the rest of the map. Thus, the
    histogram is shown in log-mode.

    Args:
        interp: matplotlib interpolation method name.
        ax:     Provide custom axis object.

    Returns:
        The axis.
    """
    if ax is None:
        fig, ax = _new_axis(**kwargs)
    ax.imshow(_np.log1p(self.whist.reshape(self.dx, self.dy)),
              vmin=0, cmap='Greys', interpolation=interp, origin='lower')
    return ax



def inspect(self):
    fig = _plt.figure(figsize=(12, 5))
    ax1 = _new_axis(sp_pos=(1, 3, 1), fig=fig)
    ax2 = _new_axis(sp_pos=(1, 3, 2), fig=fig)
    ax3 = _new_axis(sp_pos=(1, 3, 3), fig=fig)

    _, _ = self.plot_umatrix(ax=ax1)

    if self.isCalibrated:
        _ = self.plot_calibration(ax=ax2)
    else:
        _ = self.plot_whist(ax=ax2)

    self.plot_qerror(ax=ax3)


def weights(weights: Array, dims: Tuple, cmap: str = 'tab20',
        figsize: Tuple = (15, 15), stand: bool =False) -> Tuple:
    """Plot a bar chart of the weights of each map unit.

    Args:
        weights:    Two-dimensional array of weights.
        dims:       SOM dimensions (dx, dy, dw).
        cmap:       Matplotlib color map name.
        figsize:    Figure size.
        stand:      Standardize the weights if ``True``.

    Returns:
        Figure and axes.
    """
    dx, dy, dw = dims
    fig, axs = plt.subplots(dx, dy, figsize=figsize, sharex=True, sharey=True)
    axs = np.flipud(axs).flatten()

    xr = range(dw)
    bar_colors = getattr(plt.cm, cmap)(xr)
    if stand:
        weights = tools.standardize(weights)
    yticks = np.arange(np.floor(weights.min()), np.ceil(weights.max())+1, 2)

    for ax, wv in zip(axs, weights):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_xticks([])
        ax.set_yticks(yticks)
        ax.bar(xr, wv, color=bar_colors)

    return fig, axs

