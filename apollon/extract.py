"""
Copyright 2019, Michael Blaß
michael.blass@uni-hamburg.de
"""

from collections import OrderedDict as _OrderedDict

import numpy as _np

from apollon.signal.spectral import fft as _fft
from . types import Array as _Array


def maxamp(sig: _Array) -> float:
    """Extract maximal amplitude from chunk."""
    return _np.max(_np.absolute(sig))


def meanamp(sig: _Array) -> float:
    """Extract mean amplitude from chunk."""
    return _np.sum(sig) / len(sig)


def stdamp(sig: _Array) -> float:
    """Extract standard deviation from chunk."""
    return _np.std(sig)


def energy(sig: _Array) -> float:
    """Extract linear energy from chunk."""
    return _np.sum(_np.square(sig))


def entropy(sig: _Array, theta: int = 2, bins: int = 10,
            window: str = 'None') -> float:
    """Shannon Entropy in [nat], normalized by log(N).
    """
    if window:
        pps = PseudoPhaseSpace(sig * window(len(sig)), theta, bins)
    else:
        pps = PseudoPhaseSpace(sig, theta, bins)
    return pps.get_entropy()


def rms_energy(sig: _Array) -> float:
    """Extract rms energy from chunk."""
    return _np.sqrt(_np.sum(_np.square(sig)) / len(sig))


def spectral_centroid(sig: _Array, fs: int, window: str = 'hamming') -> _Array:
    """Extract spectral centroid from input array.

    Params:
        fs     (int)    Sampling frequency in Hz.
        window (str)    Window function indentifier.

    return    (numpy ndarray) of spc values.
    """
    if sig.ndim == 2:
        out = _np.zeros(sig.shape[0])

        for i, seg in enumerate(sig):
            out[i] = _fft(seg, fs=fs, window=window).centroid()
        return out

    elif sig.ndim == 1:
        return _fft(sig, fs=fs, window=window).centroid()

    else:
        raise ValueError('Input array must have at max two dimensions.')


available_features = _OrderedDict({'maxamp': maxamp,
                                   'meanamp': meanamp,
                                   'stdamp': stdamp,
                                   'entropy': entropy,
                                   'energy': energy,
                                   'rms': rms_energy,
                                   'spc': spectral_centroid,
                                   })
