#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""apollon/signal/tools.py    (c) Michael Blaß 2016

Signal processing tools

Functions:
    acf                 Normalized autocorrelation.
    acf_pearson         Normalized Pearson acf.
    amp2db              Transform amplitude to dB.
    corr_coef_pearson   Correlation coefficient after Pearson.
    freq2mel            Transform frequency to mel.
    mel2freq            Transform mel to frequency.
    loadwav             Load a .wav file.
    maxamp              Maximal amplitude of signal.
    minamp              Minimal amplitude of signal.
    noise               Generate withe noise.
    sinusoid            Generate sinusoidal signal.
    zero_padding        Append array with zeros.
"""


import numpy as _np
from scipy import stats

from apollon.signal.audio import _AudioData


def acf(inp_sig):
    """Normalized estimate of the autocorrelation function of `inp_sig`
       by means of cross correlation."""

    N = len(inp_sig)
    norm = inp_sig @ inp_sig

    out = _np.empty(N)
    out[0] = 1
    for m in range(1, N):
        a = inp_sig[:-m]
        b = inp_sig[m:]
        s = a @ b
        if s == 0:
            out[m] = 0
        else:
            out[m] = s / norm

    return out


def acf_pearson(inp_sig):
    """Normalized estimate of the autocorrelation function of `inp_sig`
       by means of pearson correlation coefficient."""

    N = len(inp_sig)
    out = _np.empty(N-1)

    out[0] = 1
    for m in range(1, N-1):

        a = inp_sig[:-m]
        b = inp_sig[m:]

        s = corr_coef_pearson(a, b)

        if s == 0:
            out[m] = 0
        else:
            out[m] = s

    return out


def amp2db(amp):
    """Transform amplitude to dB.

    Params:
        amp    (array-like or number) Given amplitude values.

    Return:
        (ndarray)    values in dB.
    """
    foo = _np.atleast_1d(amp)
    return 20 * _np.log10(foo / maxamp(foo))


def corr_coef_pearson(x, y):
    """Fast perason correlation coefficient."""
    detr_x = x - _np.mean(x)
    detr_y = y - _np.mean(y)

    foo = (detr_x @ detr_y)
    bar = (detr_x @ detr_x) * (detr_y @ detr_y)

    if bar == 0:
        return 0
    else:
        return foo / _np.sqrt(bar)


def freq2mel(freq):
    """Transforms Hz to Mel-Frequencies.

    Params:
        freq:    (real number) Frequency in Hz

    Return:
        (real number) Mel-Frequency
    """
    return 1127 * _np.log(1 + freq / 700)


def loadwav(path, norm=True):
    """Load a .wav file.

    Params:
        path    (str or fobject)
        norm    (bool) True if data should be normalized.

    Return:
        (int, ndarray)    sample rate and data.
    """
    return _AudioData(path, norm)


def mel2freq(mel):
    """Transforms Mel-Frequencies to Hz.

    Params:
        mel:     (real number) Mel-Frequency.

    Return:
        (real number) Frequency in Hz.
    """
    return 700 * (_np.exp(mel / 1125) - 1)


def maxamp(sig):
    """Maximal absolute elongation within the signal.

    Params:
        sig    (array-like) Input signal.

    Return:
        (scalar) Maximal amplitude.
    """
    return _np.max(_np.absolute(sig))


def minamp(sig):
    """Minimal absolute elongation within the signal.

    Params
        sig    (array-like) Input signal.

    Return:
        (scalar) Maximal amplitude.
    """
    return _np.min(_np.absolute(sig))


def noise(level, n=9000):
    """Generate withe noise.

    Params:
        level       (float) Noise level as standard deviation of a gaussian.
        n           (int) Length of noise signal in samples.

    Return:
        (ndarray)   White noise signal.
    """
    return stats.norm.rvs(0, level, size=n)


def sinusoid(f, amps=1, sr=9000, length=1, plot=False, retcomps=False):
    """Generate sinusoidal signal.

    Params:
        f       (iterable) Component frequencies.
        amps    (int or interable) Amplitude of each component in f.
                    If `amps` is an integer each component of f will be
                    scaled according to `amps`. If `amps` is an iterable
                    each frequency will be scaled with the respective amplitude.
        sr      (int) Sample rate.
        length  (number) Length of signal in seconds.
        plot    (bool) If True plot the signal.
        retcomps(bool) If True return the components of the signal,
                    otherwise return the sum.

    Return:
        (ndarray)   Sinusoidal signal.
    """
    f = _np.atleast_1d(f)
    amps = _np.atleast_1d(amps)

    if f.shape == amps.shape or amps.size == 1:
        t = _np.arange(sr*length)[:, None]
        f = f / sr
        sig = _np.sin(2*_np.pi*f*t) * amps
    else:
        raise ValueError('Shapes of f and amps must be equal.')

    if plot:
        plt.plot(t/sr, sig)

    if retcomps:
        return sig
    else:
        return sig.sum(axis=1)


def zero_padding(sig, n):
    """Appends n zeros to signal.

    Params:
        sig    (numerical array-like) a list of data points.
        n      (int) number of zeros to be appended.

    Return:
        (array) zero-padded input signal.
    """
    return _np.append(sig, [0] * n)
