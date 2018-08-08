#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""spectral.py    (c) Michael Blaß 2016

Provide easy access to frequency spectra obtained by the DFT.

Classes:
    _Spectrum_Base      Utility class
    _Spectrum           Representation of a frequency spectrum

Functions:
    fft                 Easy to use discrete fourier transform
"""


__author__ = 'Michael Blaß'


import numpy as _np
import matplotlib.pyplot as _plt
from scipy.signal import stft as _stft
from scipy.signal import get_window as _get_window

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

    def __abs__(self):
        return _np.absolute(self.bins)

    def abs(self):
        """Return magnitude spectrum."""
        return self.__abs__()

    def power(self):
        """Retrun power spectrum."""
        return _np.square(self.abs())

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
        window      (str) name of valid window

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
        w = _get_window(window, length)
        bins = _np.fft.rfft(sig * w, n) / length * 2
    else:
        bins = _np.fft.rfft(sig, n) / length * 2

    return _Spectrum(bins, sr, n, window)


class STFT:
    def __init__(self, sig, sr:int, window:str ='hamming', n_perseg:int=256,
                       n_overlap:int=None, n_fft:int=None):
        """Short time Fourier Transform of one-dimensional input `sig`.

        Params:
            sr          Sample rate.
            window      Window function name.
            n_perseg    Number of samples per window.
            n_overlap   Number of overlaping samples.
            n_fft       FFT length.
        """
        res = _stft(sig, fs=sr, window=window, nperseg=n_perseg,
                   noverlap=n_overlap, nfft=n_fft)

        self.freqs, self.t, self.bins = res
        self.bins *= 2
        self.shape = self.bins.shape


    def plot(self, power:bool = True):
        """Plot the spectrogram of the STFT.
        
        Params:
            power   Plot power spectrum if True.
        """
        if power:
            XX = self.power()
        else:
            XX = self.abs()

        # TODO: implement aplot integration
        fig, ax = _plt.subplots(1, figsize=(8, 4))
        ax.pcolormesh(self.t, self.freqs, XX,
                      cmap='nipy_spectral')
        return fig, ax

    def abs(self):
        return self.__abs__()

    def power(self):
        return _np.square(self.__abs__())

    def centroid(self):
        Xp = self.power()
        return _np.sum(Xp.T * self.freqs, axis=1) / _np.sum(Xp, axis=0)

    def flux(self, hr=True):
        return _np.maximum(_np.diff(self.abs(), axis=1), 0)

    def __abs__(self):
        return _np.absolute(self.bins)
