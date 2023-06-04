"""
Simple filter implementations
"""

import scipy.signal as _scs

from .. types import Array as _Array


def coef_bw_bandpass(low: int, high: int, fs: int, order: int = 4) -> tuple:
    """Return coefficients for a Butterworth bandpass filter

    Args:
        low:    Lower cutoff frequency in Hz
        high:   Upper cutoff freqency in Hz
        fs:     Sample of signal to be filtered
        order:  Order of the filter

    Returns:
        Filter coefficients
    """
    nyq = fs / 2
    b, a = _scs.butter(order, (low/nyq, high/nyq), btype='bandpass')
    return b, a


def bandpass_filter(x: _Array, fs: int, low: int, high: int, order: int = 4) -> _Array:
    """Apply a Butterworth bandpass filter to input signal ``x``

    Args:
        x:      One-dimensional input array
        fs:     Samplerate of ``x``
        low:    Lower cut-off frequency in Hz
        high:   Upper cut-off frequency in Hz
        order:  Order of the filter

    Returns:
        Filtered input signal
    """
    b, a = coef_bw_bandpass(low, high, fs, order)
    w, h = _scs.freqz(b, a)

    return _scs.lfilter(b, a, x)
