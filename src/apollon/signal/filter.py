"""
Simple filter implementations
"""
import scipy.signal as _scs

from .. typing import FloatArray, floatarray


def coef_bw_bandpass(low: int, high: int, fps: int, order: int = 4
                     ) -> tuple[FloatArray, FloatArray]:
    """Return coefficients for a Butterworth bandpass filter

    Args:
        low:    Lower cutoff frequency in Hz
        high:   Upper cutoff freqency in Hz
        fps:    Signal sample rate
        order:  Order of the filter

    Returns:
        Filter coefficients
    """
    nyq = fps / 2

    num, denom = _scs.butter(order, (low/nyq, high/nyq), btype='bandpass')
    return (floatarray(num), floatarray(denom))


def bandpass_filter(inp: FloatArray, fps: int, low: int, high: int,
                    order: int = 4) -> FloatArray:
    """Apply a Butterworth bandpass filter to input signal ``x``

    Args:
        inp:    One-dimensional input array
        fps:    Samplerate of ``x``
        low:    Lower cut-off frequency in Hz
        high:   Upper cut-off frequency in Hz
        order:  Order of the filter

    Returns:
        Filtered input signal
    """
    coeffs = coef_bw_bandpass(low, high, fps, order)
    return floatarray(_scs.lfilter(*coeffs, inp))
