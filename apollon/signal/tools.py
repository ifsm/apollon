#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""apollon/signal/tools.py    (c) Michael Blaß 2016

Signal processing tools

Functions:
    amp2db          Transform amplitude to dB.
    freq2mel        Transform frequency to mel.
    mel2freq        Transform mel to frequency.
    loadwav         Load a .wav file.
    maxamp          Maximal amplitude of signal.
    minamp          Minimal amplitude of signal.
    zero_padding    Append array with zeros.
"""


import numpy as _np

from apollon.signal.audio import _AudioData

def amp2db(amp):
    """Transform amplitude to dB.

    Params:
        amp    (array-like or number) Given amplitude values.

    Return:
        (ndarray)    values in dB.
    """
    foo = _np.atleast_1d(amp)
    return 20 * _np.log10(foo / maxamp(foo))


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


def zero_padding(sig, n):
    """Appends n zeros to signal.

    Params:
        sig    (numerical array-like) a list of data points.
        n      (int) number of zeros to be appended.

    Return:
        (array) zero-padded input signal.
    """
    return _np.append(sig, [0] * n)
