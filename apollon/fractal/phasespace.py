#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""phasespace.py (c) Michael Blaß 2016

Pseudo phase space representation.

Classes:
    PseudoPhaseSpace
"""

import numpy as _np

from apollon.signal import tools


def embedding(inp_sig, tau, m=2, mode='zero'):
    """Generate n-dimensional pseudo-phase space embedding.

    Params:
        inp_sig    (iterable) Input signal.
        tau        (int) Time shift.
        m          (int) Embedding dimensions.
        mode       (str) Either `zero` for zero padding,
                                `wrap` for wrapping the signal around, or
                                `cut`, which cuts the signal at the edges.
                         Note: In cut-mode, each dimension is only
                               len(sig) - tau * (m - 1) samples long.
    Return:
        (np.ndarray) of shape
                        (m, len(inp_sig)) in modes 'wrap' or 'zeros', or
                        (m, len(sig) - tau * (m - 1)) in cut-mode.
    """
    inp_sig = _np.atleast_1d(inp_sig)
    N = len(inp_sig)

    if mode == 'zero':
        # perform zero padding
        out = _np.zeros((m, N))
        out[0] = inp_sig
        for i in range(1, m):
            out[i, tau*i:] = inp_sig[:-tau*i]

    elif mode == 'wrap':
        # wraps the signal around at the bounds
        out = _np.empty((m, N))
        for i in range(m):
            out[i] = _np.roll(inp_sig, i*tau)

    elif mode == 'cut':
        # cut every index beyond the bounds
        Nm = N - tau * (m-1)    # number of vectors
        if Nm < 1:
            raise ValueError('Embedding params to large for input.')
        out = _np.empty((m, Nm))
        for i in range(m):
            off = N - i * tau
            out[i] = inp_sig[off-Nm:off]

    else:
        raise ValueError('Unknown mode `{}`.'.format(pad))

    return out


class PseudoPhaseSpace(_np.ndarray):
    """Pseudo phase-space of a given signal.

    This class is a representation of a 2D pseudo-phase space. It inherits
    from numpy`s ``ndarray´´ and has therefore all ndarray instance
    variables and methods available.

    Addition instance variables:
        (array) a       Input signal
        (array) b       Delayed and zero-padded signal
        (int)   bins    Number of bins per axis
        (int)   theta   Delay of b

    Additional methods:
        get_probs(self)     Return probability each bin
        get_entropy(self)   Return the entropy of the space
        plot(self, grid=False, cbar=True)   Plot the pps
    """

    def __new__(cls, signal, theta, bins, probs=False):
        '''
        :param cls:       Don't touch! Used for ndarray subclassing
        :param signal:    (array) a signal
        :param theta:     (int) a delay
        :param bins:      (int) number of boxes (== sqrt(bins))
        '''

        # Input array is an already formed ndarray instance
        a = _np.atleast_1d(signal)
        b = tools.zero_padding(signal[theta:], theta)
        data, xedges, yedges = _np.histogram2d(a, b, bins)

        if probs:
            data /= data.sum()

        # We first cast to be our class type
        obj = _np.asarray(data).view(cls)

        # add the new attributes to the created instance
        obj.a = a
        obj.b = b
        obj.bins = bins
        obj.theta = theta

        # Finally, we must return the newly created object:
        return obj

    def get_probs(self):
        """Return probabilities of the pps."""

        return _np.array(self / self.sum())

    def get_entropy(self):
        """Return the entropy of the space."""

        probs = self.get_probs()
        non_zeros = probs[_np.where(probs != 0)]
        return -_np.sum(non_zeros * _np.log(non_zeros)) / _np.log(probs.size)

    def plot(self, grid=False, cbar=True):
        """Plot the pseudo-phase space."""

        # Plot the space
        fig, ax = _plt.subplots()
        ax_im = _plt.imshow(self, aspect='auto', origin='lower',
                            vmin=0, vmax=1, interpolation='None')
        if cbar:
            _plt.colorbar(ax=ax_im)

        if grid:
            # ticklabels and their locations are the same values
            locs = _np.arange(self.bins)

            # replace ticks so that grid lines a drawn between them
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_ticks(locs + 0.5, minor=True)
                axis.set(ticks=locs, ticklabels=locs)
            _plt.grid(True, which='minor')
