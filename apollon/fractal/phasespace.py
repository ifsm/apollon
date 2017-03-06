#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""phasespace.py (c) Michael Blaß 2016

Pseudo phase space representation.

Classes:
    PseudoPhaseSpace
"""

import numpy as _np

from apollon.signal import tools


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
