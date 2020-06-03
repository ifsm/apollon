# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
"""

import scipy.signal as _scs

from .. types import Array as _Array


def coef_bw_bandpass(low: int, high: int, fs: int, order: int = 4) -> tuple:
    """Return coefficients for a Butterworth bandpass filter.

    Args:
        low   (int)    Lower cutoff frequency in Hz.
        high  (int)    Upper cutoff freqency in Hz.
        fs    (int)    Sample of signal to be filtered.
        order (int)    Order of the filter.

    Returns:
        (tuple)    (b, a) Filter coefficients.
    """
    nyq = fs / 2
    b, a = _scs.butter(order, (low/nyq, high/nyq), btype='bandpass')
    return b, a


def bandpass_filter(x: _Array, fs: int, low: int, high: int, order: int = 4) -> _Array:
    """Apply a Butterworth bandpass filter to input signal `x`.

    Args:
        x     (np.ndarray)    One-dimensional input array.
        fs    (int)           Samplerate of `x`.
        low   (int)           Lower cut-off frequency in Hz.
        high  (int)           Upper cut-off frequency in Hz.
        order (int)           Order of the filter.

    Returns:
        (np.ndarray)    Filtered input signal.
    """
    b, a = coef_bw_bandpass(low, high, fs, order)
    w, h = _scs.freqz(b, a)

    return _scs.lfilter(b, a, x)
