#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""aplot.py

(c) Michael Blaß, 2016

An implementation of advanced plotting utilities, that fits the common
Apollon workflow.

Functions:
    _new_figure         Create new (fig, ax) tuple.
    fourplot            Create a four plot of time a signal.
    marginal_distr      Plot the marginal distribution of a PoissonHMM.
    onsets              Plot onsets over a signal.
    onest_decoding      Plot decoded onsets over a signal.
    signal              Plot a time domain signal.
"""


import matplotlib.pyplot as _plt
import matplotlib.cm as _cm
import numpy as _np
from scipy import stats as _stats

from apollon.signal.audio import _AudioData
from apollon.apollon_globals import plot_params as _plot_params
from apollon.decorators import switch_interactive


def _new_figure(spines='nice', **kwargs):
    '''Create a new figure with a single axis and fancy spines
    Params:
        spines    (str) choose between "nice" and "standard" spines
        **kwargs   pass all keywords to matplotlib.figure

    Return:
        (tuple)    A figure and a AxesSubplot instance
    '''
    fig = _plt.figure(**kwargs)
    ax = fig.add_subplot(1, 1, 1)

    if spines == 'nice':
        # adjust spines positions
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))

        # Hide right and top spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Show tick on left and bottom spines, only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.axison = True
        return fig, ax

    elif spines == 'standard':
        return fig, ax


@switch_interactive
def fourplot(data, lag=1, standardized=True):
    # preprocess data
    if standardized is True:
        data = (data - _np.mean(data)) / _np.std(data)

    # equivalent to _np.roll(data, 1) but faster
    lag_data = _np.append(data[-lag:], data[:-lag])
    (osm, osr), (slope, intercept, r) = _stats.probplot(data, dist='norm')
    x = _np.arange(_np.ceil(osm[0])-1, _np.ceil(osm[-1]+1))
    regr = slope*x+intercept

    # set up the figure
    fig, ((ax1, ax2), (ax3, ax4)) = _plt.subplots(2, 2, figsize=(10, 6))
    _plt.subplots_adjust(wspace=.3, hspace=.5)

    # data as time series
    ax1.plot(data, lw=2, alpha=.5)
    ax1.set_title(r'Time series ($n$={})'.format(len(data)))
    ax1.set_xlabel('Index i')
    ax1.set_ylabel('Value')

    # lag-plot
    ax2.scatter(data, lag_data, alpha=.5)
    ax2.plot(data, lag_data, 'k', alpha=.05)
    ax2.set_title(r'Lag-plot ($\theta$={})'.format(lag))
    ax2.set_xlabel('Index i-{}'.format(lag))
    ax2.set_ylabel('Index i')

    # histogram
    counts, edges, patches = ax3.hist(data, alpha=.5, align='left')
    ax3.set_xticks(edges[:-1])
    ax3.set_xticklabels(edges[:-1])
    ax3.set_title('Normalized histogram')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')

    # probabiliti plot
    ax4.scatter(osm, osr, alpha=.5)
    ax4.plot(x, regr, 'r', lw=2, alpha=.8)
    ax4.set_title(r'Normal probability plot ($R^2$={})'
                  .format((r**2).round(4)))
    ax4.set_xlabel('Qunatiles')
    ax4.set_ylabel('Ordered values')

    return osm, osr, slope, intercept, r


@switch_interactive
def marginal_distr(x, lambda_, delta, figsize=(10, 4), bins=20, legend=True):
    '''Plot the marginal distribution of a PoissonHMM.
    Params:
        x        (array-like) time series of HMM training data
        lambda_  (array-like) componet distribution means
        delta    (array-like) stationary distribution of Poisson HMM

    Return:
        (fig, ax)   plot context.'''
    fig, ax = _new_figure(figsize=figsize)
    n, bin_edges, patches = ax.hist(x, normed=True, alpha=.2, bins=bins)

    for i, (me, de) in enumerate(zip(lambda_, delta)):
        a, b = _stats.poisson.interval(.9999, me)
        t = _np.arange(a, b, dtype=int)
        y = _stats.poisson.pmf(t, me) * de

        ax.plot(t, y, alpha=.7, lw=2, ls='dashed', label='$\lambda_{}={}$'
                .format(i, round(me, 2)))

    if legend:
        # Place legend outside the axe
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    return fig, ax


@switch_interactive
def onsets(sig, odx, figsize=(10, 4), **kwargs):
    fig, ax = signal(sig)
    ax.vlines(odx, -1, 1, **_plot_params.onset)
    return fig, ax


@switch_interactive
def onset_decoding(sig, ons, dec):
    '''Plot sig and and onsetes color coded regarding dec.'''
    fig, ax = signal(sig, params=_plot_params.sig_ons)
    lc = max(dec) + 1
    colors = _np.linspace(0, 1, lc)
    ax.vlines(ons.odx, -1, 1, linewidths=3, linestyle='dashed',
              colors=_cm.viridis(colors[dec]))
    return fig, ax


@switch_interactive
def signal(sig, xaxis='seconds', figsize=(10, 4), params=None):
    '''Plot a signal on fancy axes.'''
    fig, ax = _new_figure(figsize=figsize)

    if params is None:
        params = _plot_params.signal

    if isinstance(sig, _AudioData):
        if xaxis == 'samples':
            ax.set_xlabel('Time [samples]')
        elif xaxis == 'seconds':
            # TODO: ticklabes for signals > 1 min
            len_in_sec = sig._n / sig.get_sr()
            ticks = _np.linspace(0, sig._n, _np.ceil(len_in_sec)*2)
            ticklabels = _np.arange(0, len_in_sec, .5)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_xlabel('Time [s]')
        else:
            raise ValueError('Unknown option <{}>'.format(xaxis))
        ax.set_ylabel('Amplitude')
        ax.plot(sig.get_data(), **params)
    else:
        ax.plot(sig, **params)
    return fig, ax
