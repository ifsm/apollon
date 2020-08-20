# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/som/plot.py
"""
from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from apollon import tools
from apollon import aplot
from apollon.types import Array, Axis, Shape


def umatrix(ax: Axis, umx: Array, outline: bool = False,
            pad_mode: str = 'constant', **kwargs) -> None:
    """Plot the U-matrix.

    Args:
        ax:   Axis subplot.
        umx:  U-matrix data.

    Note:
        Figure aspect is set to 'eqaul'.
    """
    defaults = {
            'cmap': 'terrain',
            'levels': 20}
    defaults.update(kwargs)
    sdx, sdy = umx.shape
    umx_padded = np.pad(umx, 1, mode=pad_mode)

    _ = ax.contourf(umx, **defaults, extent=(-0.5, sdy-0.5, -0.5, sdx-0.5))
    _ = ax.set_xticks(range(sdy))
    _ = ax.set_yticks(range(sdx))
    if outline:
        ax.contour(umx, cmap='Greys_r', alpha=.7)
    ax.set_aspect('equal')


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


def plot_qerror(ax=None, **kwargs):
    """Plot quantization error."""
    if ax is None:
        fig, ax = _new_axis(**kwargs)

    ax.set_title('Quantization Errors per iteration')
    ax.set_xlabel('# interation')
    ax.set_ylabel('Error')

    ax.plot(self.quantization_error, lw=3, alpha=.8,
            label='Quantizationerror')




def plot_umatrix3d(w=1, cmap='viridis', **kwargs):
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
    if stand:
        weights = tools.standardize(weights)
    lower = np.floor(weights.min())
    upper = np.ceil(weights.max())
    yticks = np.linspace(lower, upper, 5)
    xr = range(dw)
    bar_colors = getattr(plt.cm, cmap)(xr)

    fig, axs = plt.subplots(dx, dy, figsize=figsize, sharex=True, sharey=True,
            subplot_kw={'xticks': [], 'yticks': yticks})
    axs = np.flipud(axs).flatten()

    for ax, wv in zip(axs, weights):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.bar(xr, wv, color=bar_colors)

    return fig, axs


def weights_line(weights: Array, dims: Tuple, color: str = 'r',
        figsize: Tuple = (15, 15), stand: bool =False) -> Tuple:
    """Plot a line chart of the weights of each map unit.

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
    if stand:
        weights = tools.standardize(weights)
    lower = np.floor(weights.min())
    upper = np.ceil(weights.max())

    fig, axs = plt.subplots(dx, dy, figsize=figsize, sharex=True, sharey=True,
            subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False})
    axs = np.flipud(axs).flatten()

    for ax, wv in zip(axs, weights):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.plot(wv, color=color)

    return fig, axs


def wire(ax: Axis, weights: Array, shape: Shape, *,
         unit_size: Union[int, float, Array] = 100.0,
         line_width: Union[int, float] = 1.0,
         highlight: Optional[Array] = None, labels: bool = False, **kwargs):
    """Plot the weight vectors of a SOM with two-dimensional feature space.

    Neighbourhood relations are indicate by connecting lines.

    Args:
        ax:          The axis subplot.
        weights:     SOM weigth matrix.
        shape:       SOM shape.
        unit_size:   Size for each unit.
        line_width:  Width of the wire lines.
        highlight:   Index of units to be marked in different color.
        labels:      If ``True``, attach a box with coordinates to each unit.

    Returns:
        vlines, hlines, bgmarker, umarker
    """
    unit_color = 'k'
    bg_color = 'w'
    hl_color = 'r'
    alpha = .7

    if isinstance(unit_size, np.ndarray):
        marker_size = tools.scale(unit_size, 10, 110)
    elif isinstance(unit_size, float) or isinstance(unit_size, int):
        marker_size = np.repeat(unit_size, weights.shape[0])
    else:
        msg = (f'Argument of parameter ``unit_size`` must be real scalar '
                'or one-dimensional numpy array.')
        raise ValueError(msg)
    marker_size_bg = marker_size + marker_size / 100 * 30

    if highlight is not None:
        bg_color = np.where(highlight, hl_color, bg_color)

    rsw = weights.reshape(*shape, 2)
    vx, vy = rsw.T
    hx, hy = np.rollaxis(rsw, 1).T
    ax.set_aspect('equal')
    vlines = ax.plot(vx, vy, unit_color, alpha=alpha, lw=line_width, zorder=9)
    hlines = ax.plot(hx, hy, unit_color, alpha=alpha, lw=line_width, zorder=9)
    bgmarker = ax.scatter(vx, vy, s=marker_size_bg, c=bg_color,
                          edgecolors='None', zorder=11)
    umarker = ax.scatter(vx, vy, s=marker_size, c=unit_color, alpha=alpha,
                         edgecolors='None', zorder=12)

    font = {'fontsize': 4,
            'va': 'bottom',
            'ha': 'center'}

    bbox = {'alpha': 0.7,
            'boxstyle': 'round',
            'edgecolor': '#aaaaaa',
            'facecolor': '#dddddd',
            'linewidth': .5,
            }

    if labels is True:
        for (px, py), (ix, iy) in zip(weights, np.ndindex(shape)):
            ax.text(px+1.3, py, f'({ix}, {iy})', font, bbox=bbox, zorder=13)

    return vlines, hlines, bgmarker, umarker


def data_2d(ax: Axis, data: Array, colors: Array,
           **kwargs) -> mpl.collections.PathCollection:
    """Scatter plot a data set with two-dimensional feature space.

    This just the usual scatter command with some reasonable defaults.

    Args:
        ax:      The axis subplot.
        data:    The data set.
        colors:  Colors for each elemet in ``data``.

    Returns:
        PathCollection.
    """
    defaults = {
            'alpha': 0.2,
            'c': colors,
            'cmap': 'plasma',
            'edgecolors': 'None',
            's': 10}
    for k, v in defaults.items():
        _ = kwargs.setdefault(k, v)
    aplot.outward_spines(ax)
    return ax.scatter(*data.T, **kwargs)
