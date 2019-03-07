"""
Copyright 2019, Michael Blaß
michael.blass@uni-hamburg.de
"""

import math as _math
from collections import OrderedDict

import numpy as _np

from apollon.signal.spectral import fft as _fft
from . types import Array as _Array


def maxamp(chunk, **kwargs):
    '''Extract maximal amplitude from chunk.'''
    return _np.max(_np.absolute(chunk))


def meanamp(chunk, **kwargs):
    '''Extract mean amplitude from chunk.'''
    return _np.sum(chunk) / len(chunk)


def stdamp(chunk, **kwargs):
    '''Extract standard deviation from chunk.'''
    return _np.std(chunk)


def energy(chunk, **kwargs):
    '''Extract linear energy from chunk.'''
    return _np.sum(_np.square(chunk))


def entropy(chunk, theta=2, bins=10, window=None, **kwargs):
    '''Shannon Entropy in [nat], normalized by log(N)'''
    if window:
        pps = PseudoPhaseSpace(chunk * window(len(chunk)), theta, bins)
    else:
        pps = PseudoPhaseSpace(chunk, theta, bins)
    return pps.get_entropy()


def rms_energy(chunk, **kwargs):
    """Extract rms energy from chunk."""
    return _math.sqrt(_np.sum(_np.square(chunk)) / len(chunk))


def spectral_centroid(inp: _Array, fs: int, window: str = 'hamming') -> _Array:
    """Extract spectral centroid from input array.

    Params:
        fs     (int)    Sampling frequency in Hz.
        window (str)    Window function indentifier.

    return    (numpy ndarray) of spc values.
    """
    if inp.ndim == 2:
        out = _np.zeros(inp.shape[0])

        for i, seg in enumerate(inp):
            out[i] = _fft(seg, fs=fs, window=window).centroid()
        return out

    elif inp.ndim == 1:
        return _fft(inp, fs=fs, window=window).centroid()

    else:
        raise ValueError('Input array must have at max two dimensions.')


available_features = OrderedDict({'maxamp': maxamp,
                                  'meanamp': meanamp,
                                  'stdamp': stdamp,
                                  'entropy': entropy,
                                  'energy': energy,
                                  'rms': rms_energy,
                                  'spc': spectral_centroid,
                                  })
