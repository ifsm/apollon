#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""spectral.py    (c) Michael Blaß 2016

Class providing easy access to frequency spectra obtained by the DFT.

Classes:
    _Spectrum_Base      Utility class
    _Spectrum           Representation of a frequency spectrum

Functions:
    fft                 Easy to use discrete fourier transform
"""


__author__ = 'Michael Blaß'


import numpy as _np
import matplotlib.pyplot as _plt

from apollon.signal.tools import amp2db


class _Spectrum_Base:
    def __abs__(self):
        return _np.absolute(self.bins)

    def __add__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins + other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins + other, sr=self.sr,
                             n=self.n, window=self.window)

    def __radd__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins + other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins + other, sr=self.sr,
                             n=self.n, window=self.window)

    def __sub__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins - other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins - other, sr=self.sr,
                             n=self.n, window=self.window)

    def __rsub__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return_Spectrum(self.bins - other.bins, sr=self.sr,
                                n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins - other, sr=self.sr,
                             n=self.n, window=self.window)

    def __mul__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins * other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins * other, sr=self.sr,
                             n=self.n, window=self.window)

    def __rmul__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins * other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins * other, sr=self.sr,
                             n=self.n, window=self.window)


class _Spectrum(_Spectrum_Base):
    def __init__(self, spectral_data, sr, n, window=None, *args, **kwargs):
        self.bins = spectral_data
        self.sr = sr
        self.n = n
        self.window = window
        self.freqs = _np.fft.rfftfreq(self.n, 1/self.sr)

    def __getitem__(self, key):
        return self.bins[key]

    def __len__(self):
        return self.length

    def __repr__(self):
        return 'Spectrum(bins={}, sr={}, n={}, window={})'.format(self.bins,
                                                                  self.sr,
                                                                  self.n,
                                                                  self.window)

    def centroid(self):
        """Return spectral centroid."""
        powspc = self.power()
        return (self.freqs * powspc).sum() / powspc.sum()

    def mag(self):
        """Return magnitude spectrum."""
        return _np.absolute(self.bins)

    def power(self):
        """Retrun power spectrum."""
        return _np.absolute(self.bins)**2

    def phase(self):
        """Return phase spectrum."""
        return _np.angle(self.bins)

    def plot(self, db=True, fmt='-', logfreq=False):
        """Plot magnitude spectrum.

        Params:
            db         (bool) set True to plot amplitudes db-scaled.
            fmt        (str) matplotlib linestyle string.
            logfreq    (bool) set True to log-scale .x axis.
        """
        fig, ax = _plt.subplots(1)

        if logfreq:
            plot_function = ax.semilogx
        else:
            plot_function = ax.plot

        if db:
            plot_function(self.freqs, 20*_np.log10(self.mag()),
                          fmt, lw=2, alpha=.7)
            ax.set_ylabel(r'Amplitude [dB]')
        else:
            plot_function(self.freqs, self.mag(),
                          fmt, lw=2, alpha=.7)
            ax.set_ylabel(r'Amplitude')

        ax.set_xlabel(r'Frequency [Hz]')
        ax.grid()

        if not _plt.isinteractive():
            fig.show()


def fft(signal, sr=None, n=None, window=None):
    """Return the discrete fourier transform of the input.

    Params:
        signal      (array-like) input time domain signal
        sr          (int) sample rate
        n           (int) fft length
        window      (function) a window function

    Returns:
        (_spectrum._Spectrum)       Spectrum object
    """
    sig = _np.atleast_1d(signal)
    length = len(sig)

    # sample rate
    if sr is None:
        sr = length
    else:
        sr = sr

    # fft length
    if n is None:
        n = length
    else:
        n = n

    if window:
        bins = _np.fft.rfft(sig * window(length), n) / length * 2
    else:
        bins = _np.fft.rfft(sig, n) / length * 2

    return _Spectrum(bins, sr, n, window)
