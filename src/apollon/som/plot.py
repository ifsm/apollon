"""apollon/som/plot.py

Plotting functions for SOMs.

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net
"""

__all__ = ['cluster_by', 'component', 'hit_counts', 'qerror', 'label_target',
           'umatrix', 'wire']

from typing import Callable, Optional, Union

import numpy as np

from apollon import tools
from apollon import aplot
from apollon.types import Array, Axis


def umatrix(ax: Axis, som, outline: bool = False, **kwargs) -> None:
    """Plot the U-matrix.

    Args:
        ax:   Axis subplot.
        som:  SOM instance.

    Note:
        Figure aspect is set to 'eqaul'.
    """
    props = {
        'cmap': 'terrain',
        'levels': 20}
    props.update(kwargs)
    _generic_contour(ax, som.umatrix(), outline, **props)


def umatrix3d(ax: Axis, som, **kwargs) -> None:
    """Plot the U-matrix in three dimensions.

    Args:
        ax:   Axis subplot.
        som:  SOM instance.

    Note:
        Figure aspect is set to 'eqaul'.
    """
    props = {
        'cmap': 'terrain',
        }
    props.update(kwargs)
    ax.plot_surface(*np.mgrid[:som.dx, :som.dy], som.umatrix(), **props)


def component(ax: Axis, som, comp: int, outline: bool = False,
              **kwargs) -> None:
    """Plot a component plane.

    Args:
        ax:    Axis subplot.
        som:   SOM instance.
        comp:  Component number.
    """
    props = {
        'cmap': 'magma',
        'levels': 20,}
    props.update(kwargs)
    _generic_contour(ax, som.weights[:, comp].reshape(som.shape), outline,
                     **props)


def label_target(ax: Axis, som, data: Array, target: Array, **kwargs) -> None:
    """Add target labels for each bmu.

    Args:
        ax:      Axis subplot.
        som:     SOM instance.
        data:    Input data.
        target:  Target labels.
    """
    props = {
        'fontsize': 9,
        'ha': 'left',
        'va': 'bottom',
        }
    props.update(kwargs)

    bmu = som.match(data)
    bmu_xy = np.fliplr(np.atleast_2d(bmu)).T
    for x, y, t in zip(*bmu_xy, target):
        ax.text(x, y, t, fontdict=props)


def qerror(ax: Axis, som, **kwargs) -> None:
    """Plot quantization error."""
    props = {
        'lw': 3,
        'alpha': .8,
        }
    props.update(kwargs)
    ax.plot(som.quantization_error, **props)


def cluster_by(ax: Axis, som, data: Array, target: Array,
               **kwargs) -> None:
    """Plot bmu colored by ``traget``.

    Args:
        ax:      Axis subplot.
        som:     SOM instance.
        data:    Input data.
        target:  Target labels.
    """
    props = {
            's': 50,
            'c': target,
            'marker': 'o',
            }
    props.update(kwargs)
    bmu = som.match(data)
    bmu_xy = np.fliplr(np.atleast_2d(bmu)).T
    ax.scatter(*bmu_xy, **props)


def hit_counts(ax: Axis, som, transform: Optional[Callable] = None,
               **kwargs) -> None:
    """Plot the winner histogram.

    Each unit is colored according to the number of times it was bmu.

    Args:
        ax:    Axis subplot.
        som:   SOM instance.
        mode:  Choose either 'linear', or 'log'.
    """
    props = {
        'interpolation': None,
        'origin': 'lower',
        'cmap': 'Greys',
        }
    props.update(kwargs)
    data = som.hit_counts.reshape(som.shape)
    if transform is not None:
        data = transform(data)
    ax.imshow(data, **props)


def wire(ax: Axis, som,
         unit_size: Union[int, float, Array] = 100.0,
         line_width: Union[int, float] = 1.0,
         highlight: Optional[Array] = None, labels: bool = False,
         **kwargs) -> None:
    """Plot the weight vectors of a SOM with two-dimensional feature space.

    Neighbourhood relations are indicate by connecting lines.

    Args:
        ax:          The axis subplot.
        som:         SOM instance.
        unit_size:   Size for each unit.
        line_width:  Width of the wire lines.
        highlight:   Index of units to be marked in different color.
        labels:      If ``True``, attach a box with coordinates to each unit.

    Returns:
        vlines, hlines, bgmarker, umarker
    """
    if isinstance(unit_size, np.ndarray):
        marker_size = tools.scale(unit_size, 10, 110)
    elif isinstance(unit_size, float) or isinstance(unit_size, int):
        marker_size = np.repeat(unit_size, som.n_units)
    else:
        msg = (f'Argument of parameter ``unit_size`` must be real scalar '
               'or one-dimensional numpy array.')
        raise ValueError(msg)
    marker_size_bg = marker_size + marker_size / 100 * 30

    bg_color: str = 'w'
    hl_color: str = 'r'

    line_props = {
        'color': 'k',
        'alpha': 0.7,
        'lw': 1.0,
        'zorder': 9,
        }
    line_props.update(kwargs)

    marker_bg_props = {
        's': marker_size_bg,
        'c': bg_color,
        'edgecolors': None,
        'zorder': 11,
        }

    marker_hl_props = {
        's': marker_size,
        'c': unit_color,
        'alpha': line_props['alpha'],
        }

    if highlight is not None:
        bg_color = np.where(highlight, hl_color, bg_color)

    rsw = som.weights.reshape(som.shape, 2)
    v_wx, v_wy = rsw.T
    h_wx, h_wy = np.rollaxis(rsw, 1).T
    vlines = ax.plot(v_wx, v_wy, **line_props)
    hlines = ax.plot(h_wx, h_wy, **line_props)
    bgmarker = ax.scatter(v_wx, v_wy, s=marker_size_bg, c=bg_color,
                          edgecolors='None', zorder=11)
    umarker = ax.scatter(v_wx, v_wy, s=marker_size, c=unit_color, alpha=alpha,
                         edgecolors='None', zorder=12)

    font = {'fontsize': 4,
            'va': 'bottom',
            'ha': 'center',
            }

    bbox = {'alpha': 0.7,
            'boxstyle': 'round',
            'edgecolor': '#aaaaaa',
            'facecolor': '#dddddd',
            'linewidth': .5,
            }

    if labels is True:
        for (px, py), (ix, iy) in zip(som.weights, np.ndindex(shape)):
            ax.text(px+1.3, py, f'({ix}, {iy})', font, bbox=bbox, zorder=13)
    ax.set_aspect('equal')
    return None



def data_2d(ax: Axis, data: Array, colors: Array,
            **kwargs) -> None:
    """Scatter plot a data set with two-dimensional feature space.

    This just the usual scatter command with some reasonable defaults.

    Args:
        ax:      The axis subplot.
        data:    The data set.
        colors:  Colors for each elemet in ``data``.

    Returns:
        PathCollection.
    """
    props = {
        'alpha': 0.2,
        'c': colors,
        'cmap': 'plasma',
        'edgecolors': 'None',
        's': 10}
    props.update(kwargs)
    aplot.outward_spines(ax)
    _ = ax.scatter(*data.T, **props)


def _generic_contour(ax: Axis, data: Array, outline: bool = False,
                     **kwargs) -> None:
    """Contour plot.

    Args:
        ax:    Axis subplot.
        data:  Two-dimensional array.
    """
    sdx, sdy = data.shape
    overwrites = {
        'extent': (-0.5, sdy-0.5, -0.5, sdx-0.5),
        }
    kwargs.update(overwrites)
    _ = ax.contourf(data, **kwargs)
    _ = ax.set_xticks(range(sdy))
    _ = ax.set_yticks(range(sdx))
    if outline:
        ax.contour(data, cmap='Greys_r', alpha=.7)
    ax.set_aspect('equal')
