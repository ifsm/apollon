"""
Signal processing tools
========================
"""
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from .. import _defaults
from .. typing import FloatArray, Int16Array, floatarray


def acf(inp: FloatArray) -> FloatArray:
    """Normalized estimate of the autocorrelation function of ``inp``
    by means of cross correlation.

    Args:
        inp:  One-dimensional input array.

    Returns:
        Autocorrelation function for all positive lags.
    """
    n_elem = len(inp)
    norm = inp @ inp
    out = np.empty(n_elem)
    out[0] = 1
    for lag in range(1, n_elem):
        pre = inp[:-lag]
        post = inp[lag:]
        prod = pre @ post
        if prod == 0:
            out[lag] = 0
        else:
            out[lag] = prod / norm
    return out


def acf_pearson(sig: FloatArray) -> FloatArray:
    """Normalized estimate of the autocorrelation function of `sig`
       by means of pearson correlation coefficient."""

    n_elem = len(sig)
    out = np.empty(n_elem-1)
    out[0] = 1
    for lag in range(1, n_elem-1):
        pre = sig[:-lag]
        post = sig[lag:]
        prod = corr_coef_pearson(pre, post)
        if prod == 0:
            out[lag] = 0
        else:
            out[lag] = prod
    return out


def corr_coef_pearson(x_sig: FloatArray, y_sig: FloatArray) -> float:
    """Compute the Pearson correlation coefficient of two signals.

    Args:
        x_sig:  One-dimensional input signal
        y_sig:  One-dimensional input signal of the same length

    Returns:
        Correlation coefficient in [-1, 1], or ``nan`` if either signal is
        constant.
    """
    x_dtr = x_sig - np.mean(x_sig)
    y_dtr = y_sig - np.mean(y_sig)
    r_xy = np.convolve(x_dtr, y_dtr[::-1], mode='valid')
    r_xx_yy = np.sqrt((x_dtr @ x_dtr) * (y_dtr @ y_dtr))
    return floatarray(np.divide(r_xy, r_xx_yy)).item()


def c_weighting(frqs: FloatArray) -> FloatArray:
    """C-weighhting for SPL.

    Args:
        frqs:    Frequencies

    Returns:
        Weight for DFT bin with center frequency ``frq``
    """
    aaa = 148693636.0
    bbb = 424.36
    sqf = np.power(frqs, 2)
    return np.divide(aaa*sqf, (sqf+aaa)*(sqf+bbb))


def hz_to_mel(frqs: float | FloatArray) -> FloatArray:
    """Transforms Hz to Mel-Frequencies.

    Args:
        frqs:  Frequencies in Hz

    Returns:
        Frequency transformed to Mel scale
    """
    frqs = np.atleast_1d(frqs)
    return floatarray(1125 * np.log(1 + frqs / 700))


def limit(inp: FloatArray, ldb: float | None = None,
          udb: float | None = None) -> FloatArray:
    """Limit the dynamic range of ``inp`` to [``ldb``, ``udb``].

    Values below the lower boundary are raised to it, and values above the
    upper boundary are lowered to it. Boundaries are given in dB SPL and
    converted to amplitudes by ``amp``, so ``inp`` holds magnitudes in Pa.
    Omit a boundary to leave that side unlimited.

    Args:
        inp:  DFT bin magnitudes in Pa
        ldb:  Lower boundary in dB SPL
        udb:  Upper boundary in dB SPL

    Returns:
        Copy of ``inp`` with its values clipped to the boundaries

    Raises:
        ValueError: If ``ldb`` exceeds ``udb``.
    """
    if ldb is not None and udb is not None and ldb > udb:
        raise ValueError(f'Lower boundary ({ldb} dB) exceeds upper boundary '
                         f'({udb} dB).')
    lth = None if ldb is None else amp(ldb).item()
    uth = None if udb is None else amp(udb).item()
    return floatarray(np.clip(inp, lth, uth))


def mel_to_hz(zfrq: float | FloatArray) -> FloatArray:
    """Transforms Mel-Frequencies to Hzfrq.

    Args:
        zfrq:  Mel-Frequencies

    Returns:
        Frequency in Hz.
    """
    zfrq = np.atleast_1d(zfrq)
    out = np.empty_like(zfrq, dtype=np.float64)
    np.exp(zfrq / 1125, out=out)
    np.subtract(out, 1, out=out)
    np.multiply(out, 700, out=out)
    return out


def maxamp(sig: FloatArray) -> FloatArray:
    """Maximal absolute elongation within the signal.

    Args:
        sig: Input signal

    Returns:
        Maximal amplitude
    """
    val: FloatArray = np.absolute(sig, dtype=np.double).max(axis=0)
    return val


def minamp(sig: FloatArray) -> FloatArray:
    """Minimal absolute elongation within the signal.

    Args:
        sig: Input signal

    Returns:
        Minimal amplitude
    """
    val: FloatArray = np.absolute(sig, dtype=np.double).min(axis=0)
    return val


def white_noise(level: float, n_samples: int = 9000) -> FloatArray:
    """Generate withe noise.

    Args:
        level:      Noise level as standard deviation of Gaussian
        n_samples:  Length of noise signal in samples

    Returns:
        White noise signal
    """
    return np.random.normal(0, level, n_samples)
    # return stats.norm.rvs(0, level, size=n_samples, dtype=np.double)


def normalize(sig: FloatArray) -> FloatArray:
    """Normalize a signal to [-1.0, 1.0].

    Each channel is divided by its own peak amplitude. Silent channels have
    no peak and are returned as zeros.

    Args:
        sig: Input signal

    Return:
        Normalized signal
    """
    peak = maxamp(sig)
    return floatarray(sig / np.where(peak > 0, peak, 1.0))


def sinusoid(frqs: Sequence[float] | float,
             amps: Sequence[float] | float = 1,
             fps: int = 9000, length: float = 1.0,
             noise: float | None = None, comps: bool = False) -> FloatArray:
    # pylint: disable = R0913
    """Generate sinusoidal signal.

    Args:
        frqs:    Component frequencies.
        amps:    Amplitude of each component in ``frqs``. If ``amps`` is an
                 integer, each component of ``frqs`` is scaled according to
                 ``amps``. If ``amps`` iis an iterable each frequency is scaled
                 by the respective amplitude.
        fps:     Sample rate.
        length:  Length of signal in seconds.
        noise:   Add gaussian noise with standard deviation ``noise`` to each
                 sinusodial component.
        comps:   If True, return the components of the signal,
                 else return the sum.

    Return:
        FloatArray of signals.
    """
    frqs_: FloatArray = np.atleast_1d(frqs).astype(np.double)
    amps_: FloatArray = np.atleast_1d(amps).astype(np.double)

    size = np.ceil(fps*length).astype(np.int64)
    out = np.empty((size, len(frqs_)), dtype=np.double)
    if frqs_.shape == amps_.shape or amps_.size == 1:
        txs = np.arange(fps*length)[:, None] / fps
        np.sin(2*np.pi*txs*frqs_, out=out)
        np.multiply(out, amps_, out=out)
    else:
        raise ValueError(f'Shape of ``frqs`` ({frqs_.shape}) differs from shape '
                         f' of ``amps``({amps_.shape}).')
    if noise:
        out += stats.norm.rvs(0, noise, size=out.shape)

    if comps:
        return out
    total = np.empty((out.shape[0], 1))
    np.sum(out, axis=1, keepdims=True, out=total)
    return total


def ampmod(frq_c: float, frq_m: float, mod: float, amp_c: float = 0.5,
           fps: int = 9000, length: float = 1.0) -> FloatArray:
    r"""Generate amplitude modulated sinusoids

    The modulation index `mod` is defined by

    .. math::

         m = \frac{a_{m}}{a_{c}} \,

    and determines the influcence of the modulator on the carrier. For
    incoherent demodultaion, `mod` should range in [0, 1[, where `mod` = 0 means no
    modulation.

    Args:
        frq_c:  Carrier frequency
        frq_m:  Modulator frequency
        mod:    Modulation index
        amp_c:  Carrier amplitude
        fps:    Sample rate
        length: Length of the resulting signal in seconds

    Returns:
        Modulated signal
    """
    txs = np.arange(fps*length, dtype=np.double)[:, None] / fps
    out = np.empty_like(txs, dtype=np.double)
    wts = 2 * np.pi * txs
    f_sb1 = frq_c - frq_m
    f_sb2 = frq_c + frq_m

    np.subtract(np.cos(f_sb1*wts), np.cos(f_sb2*wts), out=out)
    np.multiply(out, mod/2, out=out)
    np.add(out, np.sin(frq_c*wts), out=out)
    np.multiply(out, amp_c, out=out)
    return out


def amp(spl: Sequence[float] | float,
        ref: float = _defaults.SPL_REF) -> FloatArray:
    """Computes amplitudes form sound pressure level.

    The reference pressure defaults to the human hearing
    treshold of 20 μPa.

    Args:
        spl:    Sound pressure level

    Returns:
        DFT magnituds
    """
    return np.power(10.0, 0.05*np.atleast_1d(spl)) * ref


def _assert_positive_int(val: int, name: str) -> None:
    """Raise if ``val`` is not a positive integer.

    Integer types other than ``int`` are accepted, the integer scalars of
    ``numpy`` in particular.

    Args:
        val:   Value to test
        name:  Parameter name to refer to in the error message

    Raises:
        TypeError:  If ``val`` is not of integer type
        ValueError: If ``val`` is less than one
    """
    if not isinstance(val, numbers.Integral):
        raise TypeError(f'Argument to ``{name}`` must be an integer. '
                        f'Found {type(val)}.')
    if val < 1:
        raise ValueError(f'Argument to ``{name}`` must be positive. '
                         f'Found {val}.')


def _ms_to_frames(fps: int, duration: int, name: str) -> int:
    """Convert a duration in milliseconds to a number of frames.

    Fractional frames are rounded half up, so durations shorter than half a
    frame convert to zero frames.

    Args:
        fps:       Number of frames per second
        duration:  Duration in milliseconds
        name:      Parameter name to refer to in the error message

    Returns:
        Number of frames covered by ``duration``

    Raises:
        TypeError:  If ``duration`` is not of integer type
        ValueError: If ``duration`` is less than one
    """
    _assert_positive_int(duration, name)
    return (fps*duration + 500) // 1000


def trim_ms(sig: FloatArray, fps: int, pre: int | None = None,
            post: int | None = None) -> FloatArray:
    """Trim the given durations from the start and the end of ``sig``.

    Durations are given in milliseconds and converted to frames given the
    frame rate ``fps``, rounding half up. Hence, a duration shorter than half
    a frame trims nothing. Omit a boundary to leave that end untouched; at
    least one of ``pre`` and ``post`` has to be given.

    ``fps``, ``pre``, and ``post`` must be positive integers. They are
    annotated as ``int`` for the benefit of static type checking, but any
    integer type is accepted at runtime, the integer scalars of ``numpy`` in
    particular.

    The input signal must be two-dimensional with shape
    ``(n_frames, n_channels)``. Trimming is applied along the time axis, so
    all channels are trimmed alike.

    Args:
        sig:   Two-dimensional input signal
        fps:   Number of frames per second
        pre:   Duration to trim from the start in milliseconds
        post:  Duration to trim from the end in milliseconds

    Returns:
        View of ``sig`` with the respective ends trimmed

    Raises:
        ValueError: If ``sig`` is not two-dimensional, if neither ``pre`` nor
                    ``post`` is given, if ``fps``, ``pre``, or ``post`` is not
                    positive, or if the requested durations leave no frames
        TypeError:  If ``fps``, ``pre``, or ``post`` is not of integer type
    """
    if sig.ndim != 2:
        raise ValueError(f'Input array has {sig.ndim} dimensions. However,'
                         ' ``trim_ms`` expects two-dimensional array.')
    if pre is None and post is None:
        raise ValueError('At least one of ``pre`` and ``post`` must be given.')
    _assert_positive_int(fps, 'fps')
    n_pre = 0 if pre is None else _ms_to_frames(fps, pre, 'pre')
    n_post = 0 if post is None else _ms_to_frames(fps, post, 'post')
    stop = sig.shape[0] - n_post
    if n_pre >= stop:
        raise ValueError(f'Trimming {n_pre} frames from the start and '
                         f'{n_post} frames from the end of a signal of '
                         f'{sig.shape[0]} frames leaves nothing.')
    return sig[n_pre:stop]


def zero_padding(sig: FloatArray, n_pad: int,
                 dtype: str | np.dtype[Any] | None = None) -> FloatArray:
    """Append n zeros to signal. `sig` must be 1D array.

    Args:
        sig:    FloatArray to be padded.
        n_pad:  Number of zeros to be appended.

    Returns:
        Zero-padded input signal.
    """
    if dtype is None:
        dtype = sig.dtype
    container = np.zeros(sig.size+n_pad, dtype=dtype)
    container[:sig.size] = sig
    return container


def fti16(inp: FloatArray) -> Int16Array:
    """Cast audio loaded as float to int16.

    Args:
        inp:    Input array of dtype float64.

    Returns:
        Array of dtype int16.
    """
    vals = np.clip(np.floor(inp*2**15), -2**15, 2**15-1)
    return np.asarray(vals).astype('int16')
