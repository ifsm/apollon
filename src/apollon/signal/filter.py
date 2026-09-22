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
        high:   Upper cutoff frequency in Hz
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


def triangular_filter_bank(frqs: FloatArray, low: float, high: float, n_filters: int,
                           scale: Literal["mel", "hz"] = "mel"
                          ) -> FloatArray:
    """Compute a bank of triangular filters on a given frequency axis.

    This function computes ``n_filters`` triangular filters whose centre
    frequencies are linearly spaced on the given frequency scale. The Mel
    scale ("mel") yields the perceptually warped bank used for cepstral
    coefficients, whose filters are narrow at low and wide at high
    frequencies. Spacing them in Hertz ("hz") yields a constant-bandwidth
    bank, every filter of the same width.

    The filters are evaluated at the frequencies in ``frqs`` themselves, so
    each apex sits on its exact centre frequency, and ``frqs`` need not be
    uniformly spaced. Following the convention of this package, ``frqs`` is
    the ``Nx1`` frequency axis as returned by ``spectral.Stft.frqs``; a plain
    one-dimensional axis is accepted, too.

    Each filter spans from the previous filter's centre to the next one's and
    peaks at unity, so neighbouring filters overlap by half and their
    responses sum to one everywhere between the first and the last centre
    frequency. Note that the peak of a filter narrower than the spacing of
    ``frqs`` is not sampled exactly, in which case the largest weight in its
    row falls short of one.

    Args:
        frqs:       Frequency axis in Hz, shaped ``(N,)`` or ``(N, 1)``
        low:        Lower cut-off frequency in Hz
        high:       Upper cut-off frequency in Hz
        n_filters:  Number of filters
        scale:      Frequency scale to space the filters on, "mel" or "hz"

    Returns:
        Array of ``n_filters`` rows and one column per element of ``frqs``.

    Raises:
        ValueError: If ``frqs`` is not a single axis of at least one
            frequency, if ``low`` is negative, if ``low`` is not less than
            ``high``, if ``high`` exceeds the largest frequency in ``frqs``,
            if ``n_filters`` is less than one, if ``scale`` is unknown, or if
            ``frqs`` resolves too few filters, which would leave a filter
            without a single frequency to act on.
    """
    if frqs.ndim > 2 or (frqs.ndim == 2 and frqs.shape[1] != 1):
        raise ValueError("``frqs`` is not a single frequency axis. Expected "
                         f"shape (N,) or (N, 1), got {frqs.shape}")

    axis = np.asarray(frqs, dtype=np.double).ravel()
    if axis.size == 0:
        raise ValueError("``frqs`` is empty")

    if low < 0:
        raise ValueError("Lower cut-off frequency below 0 Hz")

    if low >= high:
        raise ValueError("Lower cut-off frequency greater or equal then high")

    if high > axis.max():
        raise ValueError("Upper cut-off frequency greater than the highest "
                         "frequency of ``frqs``")

    if n_filters < 1:
        raise ValueError("``n_filters`` is less than one")

    if scale == "mel":
        edges = mel_space(low, high, n_filters+2, endpoint=True).ravel()
    elif scale == "hz":
        edges = np.linspace(low, high, n_filters+2, endpoint=True)
    else:
        raise ValueError(f"Unknown frequency scale {scale!r}")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("``n_filters`` is too large for the given cut-off "
                         f"frequencies. Their spacing on the {scale!r} scale "
                         "collapses.")

    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (axis-lower) / (center-lower)
    falling = (upper-axis) / (upper-center)
    fbank = np.maximum(0.0, np.minimum(rising, falling))

    empty = np.flatnonzero(~fbank.any(axis=1))
    if empty.size:
        raise ValueError(f"{empty.size} of {n_filters} filters are narrower "
                         "than the spacing of ``frqs`` and hence act on no "
                         "frequency at all. Reduce ``n_filters`` or refine "
                         "``frqs``.")

    return floatarray(fbank)


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
