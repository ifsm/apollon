"""apollon/aplot.py

General plotting routines.

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net

Functions:
    fourplot            Create a four plot of time a signal.
    marginal_distr      Plot the marginal distribution of a PoissonHMM.
    onsets              Plot onsets over a signal.
    onest_decoding      Plot decoded onsets over a signal.
    signal              Plot a time domain signal.
"""
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as _plt
import matplotlib.cm as _cm
import numpy as _np
from scipy import stats as _stats

from . import _defaults
from . import tools as _tools
from . types import Array as _Array, Axis


Limits = Optional[Tuple[int, int]]
MplFig = Optional[_plt.Figure]
FigSize = Tuple[float, float]
SubplotPos = Optional[Tuple[int, int, int]]
Axes = Union[Axis, Iterable[Axis]]


def outward_spines(axs: Axes, offset: float = 10.0) -> None:
    """Display only left and bottom spine and displace them.

    Args:
        axs:     Axis or iterable of axes.
        offset:  Move the spines ``offset`` pixels in the negative direction.

    Note:
        Increasing ``offset`` may breaks the layout. Since the spine is moved,
        so is the axis label, which is in turn forced out of the figure's
        bounds.
    """
    for ax in _np.atleast_1d(axs).ravel():
        ax.spines['left'].set_position(('outward', offset))
        ax.spines['bottom'].set_position(('outward', offset))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def center_spines(axs: Axes,
                  intersect: Tuple[float, float] = (0.0, 0.0)) -> None:
    """Display axes in crosshair fashion.

    Args:
        axs:        Axis or iterable of axes.
        intersect:  Coordinate of axes' intersection point.
    """
    for ax in _np.atleast_1d(axs).ravel():
        ax.spines['left'].set_position(('axes', intersect[0]))
        ax.spines['bottom'].set_position(('axes', intersect[1]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def _new_axis(spines: str = 'nice', fig: MplFig = None, sp_pos: SubplotPos = None,
              axison: bool = True, **kwargs) -> tuple:
    """Create a new figure with a single axis and fancy spines.

    All ``kwargs`` are passed on to _plt.figure().

    Args:
        spines:          Plot mode for spines. Either 'nice' or 'standard'.
        fig:             Existing figure.
        sp_pos:          Position of the axis in the figure.
        axison:          Draw spines if True.

    Returns:
        Figure and axes.
    """

    # pylint: disable=too-many-arguments

    if 'figsize' not in kwargs:
        kwargs['figsize'] = (10, 4)

    fig = _plt.figure(**kwargs) if fig is None else fig

    if sp_pos is None:
        sp_pos = (1, 1, 1)
    ax = fig.add_subplot(*sp_pos)

    if not axison:
        ax.axison = False
    elif spines == 'nice':
        _nice_spines(ax, offset=10)

    _plt.subplots_adjust(top=.95, bottom=.15)
    return fig, ax


def _new_axis_3d(fig: MplFig = None, **kwargs) -> tuple:
    """Create a 3d cartesian coordinate system.

    Args:
        fig:    Place the new Axes3d object in ``fig``.
                If fig is ``None``, a new figure is created.

    Returns:
        Figure and axes.
    """
    fig = _plt.figure(**kwargs) if fig is None else fig
    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
    return fig, ax_3d


def signal(values: _Array, fps: int = None, **kwargs) -> tuple:
    """Plot time series with constant sampling interval.

    Args:
        values:        Values of the time series.
        fps:           Sampling frequency in samples.
        time_scale:    Seconds or samples.

    Returns:
        Figure and axes.
    """
    fig, ax = _new_axis(**kwargs)
    domain = _np.arange(values.size, dtype='float64')

    if fps is None:
        ax.set_xlabel('n [samples]')
        ax.set_ylabel(r'x[$n$]')
    else:
        domain /= float(fps)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(r'x[$t$]')

    ax.plot(domain, values, **_defaults.PP_SIGNAL)

    return fig, ax


def fourplot(data: _Array, lag: int = 1) -> tuple:
    """Plot time series, lag-plot, histogram, and probability plot.

    Args:
        data:    Input data set.
        lag:     Lag for lag-plot given in number of samples.

    Returns:
        Parameters
    """

    # pylint: disable=invalid-name

    data = _tools.standardize(data)
    (osm, osr), (slope, intercept, r) = _stats.probplot(data, dist='norm')
    x_scale = _np.arange(_np.ceil(osm[0])-1, _np.ceil(osm[-1]+1))
    regr = slope * x_scale + intercept

    # set up the figure
    _, ((ax1, ax2), (ax3, ax4)) = _plt.subplots(2, 2, figsize=(10, 6))
    _plt.subplots_adjust(wspace=.3, hspace=.5)

    # data as time series
    ax1.plot(data, lw=2, alpha=.5)
    ax1.set_title(r'Time series ($N$={})'.format(data.size))
    ax1.set_xlabel('i')
    ax1.set_ylabel(r'x[$i$]')

    # lag-plot
    ax2.scatter(data[:-lag], data[lag:], alpha=.5)
    ax2.plot(data[:-lag], data[lag:], 'k', alpha=.05)
    ax2.set_title(r'Lag plot ($ \ \theta$={})'.format(lag))
    ax2.set_xlabel(r'x[$i$]')
    ax2.set_ylabel(r'x[$i-\theta$]')

    # histogram
    ax3.hist(data, alpha=.5, align='mid')
    ax3.set_title('Histogram')
    ax3.set_xlabel('bins')
    ax3.set_ylabel('Number of samples per bin')

    # probability plot
    ax4.scatter(osm, osr, alpha=.5)
    ax4.plot(x_scale, regr, 'r', lw=2, alpha=.8)
    ax4.set_title(r'Normal probability plot ($R^2$={})'
                  .format((r**2).round(4)))
    ax4.set_xlabel('Qunatiles')
    ax4.set_ylabel('Sorted values')

    return osm, osr, slope, intercept, r


def marginal_distr(train_data: _Array, state_means: _Array, stat_dist: _Array, bins: int = 20,
                   legend: bool = True, **kwargs) -> tuple:
    """Plot the marginal distribution of a PoissonHMM.

    Args:
        train_data:     Training data set.
        state_means:    State dependend means.
        stat_dist:      Stationary distribution.

    Returns:
        Figure and Axes.
    """

    # pylint: disable=too-many-arguments, too-many-locals

    _, ax = _new_axis(**kwargs)
    _ = ax.hist(train_data, normed=True, alpha=.2, bins=bins)

    for i, (mean_val, stat_prob) in enumerate(zip(state_means, stat_dist)):
        lower, upper = _stats.poisson.interval(.9999, mean_val)
        support = _np.arange(lower, upper, dtype=int)
        prob_mass = _stats.poisson.pmf(support, mean_val) * stat_prob

        plot_label = r'$\lambda_{}={}$'.format(i, round(mean_val, 2))
        _ = ax.plot(support, prob_mass, alpha=.7, lw=2, ls='dashed', label=plot_label)

    if legend:
        # Place legend outside the axe
        _ = ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    return ax


def onsets(sig, ons, **kwargs) -> tuple:
    """Indicate onsets on a time series.

    Args:
        sig:    Input to onset detection.
        ons:    Onset detector instance.

    Returns:
        Figure and axes.
    """
    fig, ax = signal(sig.data, fps=None, **kwargs)
    odf_domain = _np.linspace(ons.n_perseg // 2, ons.hop_size * ons.odf.size,
                              ons.odf.size)
    ax.plot(odf_domain, ons.odf/ons.odf.max(), alpha=.8, lw=2)
    ax.vlines(ons.index(), -1, 1, colors='C1', lw=2, alpha=.8)
    return fig, ax


def onset_decoding(odf: _Array, onset_index: _Array, decoding: _Array,
                   cmap='viridis', **kwargs) -> tuple:
    """Plot sig and and onsetes color coded regarding dec.

    Args:
        odf:            Onset detection function or an arbitrary time series.
        onset_index:    Onset indices relative to ``odf``.
        decoding:       State codes in [0, ..., n].
        cmap:           Colormap for onsets.

    Returns:
        Figure and axes.
    """
    fig, ax = onsets(odf, onset_index, **kwargs)
    color_space = getattr(_cm, cmap)(_np.linspace(0, 1, decoding.max()+1))
    ax.vlines(onset_index, -1, 1, linewidths=3, linestyle='dashed',
              colors=color_space(decoding))
    return fig, ax
