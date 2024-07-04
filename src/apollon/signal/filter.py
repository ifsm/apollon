"""
Simple filter implementations
"""
from typing import Literal, Self, Sequence

import numpy as np
from pydantic import BaseModel, model_validator
import scipy.signal as _scs

from .. typing import FloatArray, floatarray
from . tools import mel_to_hz, hz_to_mel


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


def triang_filter_bank(low: float, high: float, n_filters: int, fps: int, size: int,
                       domain: Literal["mel"] = "mel"
                      ) -> FloatArray:
    """Compute a bank of triangular filters.

    This function computes ``n_filters`` triangular filters. The center
    frequencies are linearly spaced in the given domain. Currently, only
    'Mel' domain is implemented.

    Args:
        low:        Lower cut-off frequency in Hz
        high:       Upper cut-off frequency in Hz
        n_filters:  Number of filters
        fps:        Sample rate
        n_fft:      FFT length
        domain:     Spacing domain, either "mel", "hz". Default ist "mel".

    Returns:
        Array with ``n_filters`` rows and columns determined by ``n_fft``.
    """
    if low < 0:
        raise ValueError("Lower cut-off frequency below 0 Hz")

    if low >= high:
        raise ValueError("Lower cut-off frequency greater or equal then high")

    if high > fps//2:
        raise ValueError("Upper cut-off frequency greater or equal Nyquist")

    frq_space = mel_space(low, high, n_filters+2, endpoint=True)
    filter_frqs = np.lib.stride_tricks.sliding_window_view(frq_space.ravel(), 3)
    return triang(fps, size, filter_frqs)


def mel_space(start: float, stop: float, num: int, endpoint: bool = True) -> FloatArray:
    space = np.linspace(hz_to_mel(start), hz_to_mel(stop), num, endpoint=endpoint)
    return mel_to_hz(space)


def bin_from_frq(fps: int, size: int, frqs: float | FloatArray) -> FloatArray:
    """Compute the index of the FFT bin with closest center frequency to ``frqs``.

    This function computes the bin index regarding a real FFT.

    Args:
        fps: Sample rate
        n_fft: FFT length
        frqs: Frequencies in Hz

    Returns:
        Index of nearest FFT bin.
    """
    out = np.empty_like(frqs, dtype=int)
    np.rint(frqs*size/fps, casting="unsafe", out=out)
    return out


def triang(fps: int, n_fft: int, frqs: FloatArray,
           amps: tuple[float, float, float] = (0.0, 1.0, 0.0)
           ) -> FloatArray:
    """Compute a triangular filter.

    Compute a triangular filter of size ``n_fft'' from an array of frequencies
    ``frqs''.  The frequency array must be of shape (n, 3), where each row
    corresponds to a filter and the columns are interpreted as the lower
    cut-off, center, and upper cut-off frequencies.

    The filter response at the constituting frequencies is controlled with a
    triplet of amplitudes ``amps''. The default specifies no response at the
    cut-off frequencies and maximal response at the center frequency.

    The filter has zero response at each of the remaining frequencies.

    Args:
        fps:    Sampling rate
        n_fft:   Length of the filter
        frqs:   Constituting freqencies
        amps:   Amplitude of the filter at the constituting frequencies

    Returns:
        Array of triangular filters with shape (frqs.shape[0], size).
    """
    if n_fft < 4:
        raise ValueError("``n_fft'' is less than 3")

    filters = []
    for low, ctr, high in bin_from_frq(fps, n_fft, frqs):
        out = np.zeros((n_fft+1)//2 if n_fft % 2 else n_fft//2+1)
        roi = np.arange(low, high+1, dtype=int)
        out[roi] = np.interp(roi, (low, ctr, high), amps)
        filters.append(out)
    return np.vstack(filters)


class TriangFilterSpec(BaseModel):
    low: float
    high: float
    n_filters: int

    @model_validator(mode="after")
    def check_low_lt_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low freq must be less then high")
        return self
