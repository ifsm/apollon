"""
Signal processing tools
"""
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from .. import _defaults
from .. types import FloatArray, floatarray


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
    """Fast perason correlation coefficient."""
    x_dtr = x_sig - np.mean(x_sig)
    y_dtr = y_sig - np.mean(y_sig)
    r_xy = np.convolve(x_dtr, y_dtr[::-1], mode='valid')
    r_xx_yy = (x_dtr @ x_dtr) * (y_dtr @ y_dtr)
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


def freq2mel(frqs: FloatArray) -> FloatArray:
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
    """Limit the dynamic range of ``inp`` to  [``ldb``, ``udb``].

    Boundaries are given in dB SPL.

    Args:
        inp:  DFT bin magnitudes
        ldb:  Lower clip boundary in deci Bel
        udb:  Upper clip boundary in deci Bel

    Returns:
        Copy of ``inp`` with values clipped
    """
    if ldb is None:
        lth = 0.0
    elif isinstance(ldb, (int, float)):
        lth = amp(ldb).item()
    else:
        msg = (f'Argument to ``ldb`` must be of types ``int``, or ``float``.\n'
               f'Found {type(ldb)}.')
        raise TypeError(msg)

    if udb is None:
        uth = 0.0
    elif isinstance(udb, (int, float)):
        uth = inp.max()
    else:
        msg = (f'Argument to ``udb`` must be of types ``int``, or ``float``.\n'
               f'Found {type(ldb)}.')
        raise TypeError(msg)

    low = np.where(inp < lth, 0.0, inp)
    return np.minimum(low, uth)


def mel2freq(zfrq: FloatArray) -> FloatArray:
    """Transforms Mel-Frequencies to Hzfrq.

    Args:
        zfrq:  Mel-Frequencies

    Returns:
        Frequency in Hz.
    """
    zfrq = np.atleast_1d(zfrq)
    out = np.empty_like(zfrq)
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
    val: FloatArray = np.absolute(sig, dtype=np.float64).max(axis=0)
    return val


def minamp(sig: FloatArray) -> FloatArray:
    """Minimal absolute elongation within the signal.

    Args:
        sig: Input signal

    Returns:
        Minimal amplitude
    """
    val: FloatArray = np.absolute(sig, dtype=np.float64).min(axis=0)
    return val


def noise(level: float, n_samples: int =9000) -> FloatArray:
    """Generate withe noise.

    Args:
        level:      Noise level as standard deviation of Gaussian
        n_samples:  Length of noise signal in samples

    Returns:
        White noise signal
    """
    return np.random.normal(0, level, n_samples)
    # return stats.norm.rvs(0, level, size=n_samples, dtype=np.float64)


def normalize(sig: FloatArray) -> FloatArray:
    """Normlize a signal to [-1.0, 1.0].

    Args:
        sig: Input signal

    Return:
        Normalized signal
    """
    return sig / maxamp(sig)


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
    frqs_: FloatArray = np.atleast_1d(frqs).astype(np.float64)
    amps_: FloatArray = np.atleast_1d(amps).astype(np.float64)

    size = np.ceil(fps*length).astype(np.int64)
    out = np.empty((size, len(frqs_)), dtype=np.float64)
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
