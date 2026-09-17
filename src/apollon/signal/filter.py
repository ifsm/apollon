"""
Simple filter implementations
"""
from typing import Literal

import numpy as np
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

    if domain == "mel":
        frq_space = mel_space(low, high, n_filters+2, endpoint=True)
    filter_frqs = np.lib.stride_tricks.sliding_window_view(frq_space.ravel(), 3) # pylint: disable=[E0606]
    return triang(fps, size, filter_frqs)


def mel_space(start: float, stop: float, num: int, endpoint: bool = True) -> FloatArray:
    """Compute evenly spaced values in Mel space.

    Args:
        start:      Starting value of the sequence.
        stop:       The end value of the sequence.
        num:        Number of values to generate.
        endpoint:   If ``True``, include ``stop``. Default ``True``.

    Returns:
        Array of linearly spaced Mel values.
    """
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


def preemphasis(inp: FloatArray, coef: float = 0.97,
                prev: float | FloatArray | None = None
                ) -> tuple[FloatArray, FloatArray]:
    """Apply a first-order pre-emphasis filter to ``inp``.

    The filter computes ``out[n] = inp[n] - coef * inp[n-1]``, boosting the
    high-frequency content of the signal. Following the convention of this
    package, time runs along the first axis: an array of shape
    ``(n_samples, n_channels)`` is filtered per channel. One-dimensional
    input is treated as a single channel.

    The sample preceding ``inp[0]`` is unknown. By default it is linearly
    extrapolated as ``2*inp[0] - inp[1]``, which requires at least two
    samples. Pass ``prev`` to supply it explicitly. Filtering a signal in
    successive blocks is therefore equivalent to filtering it in one go,
    provided each call receives the state returned by its predecessor.

    Args:
        inp:    Input array, filtered along its first axis
        coef:   Pre-emphasis coefficient
        prev:   Sample preceding ``inp[0]``. Pass the state returned by the
                previous call when filtering successive blocks. If ``None``,
                it is linearly extrapolated from ``inp``.

    Returns:
        Filtered signal, and the state to pass as ``prev`` on the next call.

    Raises:
        ValueError: If ``inp`` is empty, or if it holds less than two samples
            and no ``prev`` is given.
    """
    if inp.shape[0] < 1:
        raise ValueError("``inp`` is empty along its first axis")

    if prev is None:
        if inp.shape[0] < 2:
            raise ValueError("``inp`` holds less than two samples. Cannot "
                             "extrapolate the preceding sample. Pass "
                             "``prev`` explicitly.")
        prev = 2 * inp[0] - inp[1]

    out = np.empty_like(inp, dtype=np.double)
    np.subtract(inp[1:], coef*inp[:-1], out=out[1:])
    out[0] = inp[0] - coef*prev
    return out, floatarray(inp[-1])


def lifter(inp: FloatArray, lift: float) -> FloatArray:
    """Apply a sinusoidal lifter to cepstral coefficients.

    The lifter scales the cepstral coefficient with index ``n`` by

        ``w[n] = 1 + lift/2 * sin(pi*n/lift)``,

    which attenuates the low-order coefficients relative to the higher ones,
    equalizing their otherwise widely differing numerical ranges.

    Following the convention of this package, the coefficients run along the
    first axis: an array of shape ``(n_coefs, n_frames)`` is liftered per
    frame. Counting starts at one, so the first coefficient -- usually the DC
    term ``c_0`` -- is scaled by ``w[1]`` rather than passed through
    unchanged. This matches the convention used by ``librosa``.

    ``w[n]`` turns negative for ``n > lift``, flipping the sign of the
    affected coefficients. Pick ``lift`` at least as large as the number of
    coefficients, such as the customary ``lift=22`` for 13 coefficients.

    Args:
        inp:    Array of cepstral coefficients, liftered along its first axis
        lift:   Liftering parameter. ``0`` leaves ``inp`` unchanged

    Returns:
        Liftered coefficients

    Raises:
        ValueError: If ``lift`` is negative
    """
    if lift < 0:
        raise ValueError("``lift`` is negative")

    idx = np.arange(1, inp.shape[0]+1)
    if lift == 0:
        win = np.ones_like(idx, dtype=np.double)
    else:
        win = 1 + lift/2 * np.sin(np.pi*idx/lift)
    return floatarray(inp * win.reshape((-1,) + (1,)*(inp.ndim-1)))
