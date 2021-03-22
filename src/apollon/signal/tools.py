"""apollon/signal/tools.py

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net

Functions:
    acf                 Normalized autocorrelation.
    acf_pearson         Normalized Pearson acf.
    corr_coef_pearson   Correlation coefficient after Pearson.
    c_weighting         C-weighting for SPL.
    freq2mel            Transform frequency to mel.
    limit               Limit dynamic range.
    mel2freq            Transform mel to frequency.
    frq2bark            Transform frequency to Bark scale.
    maxamp              Maximal amplitude of signal.
    minamp              Minimal amplitude of signal.
    normalize           Scale data betwee -1.0 and 1.0.
    noise               Generate white noise.
    sinusoid            Generate sinusoidal signal.
    zero_padding        Append array with zeros.
    trim_spectrogram    Trim spectrogram to a frequency range.
"""

import numpy as np
from scipy import stats

from .. import _defaults
from .. types import Array, Optional, Sequence, Union


def acf(inp: Array) -> Array:
    """Normalized estimate of the autocorrelation function of ``inp``
    by means of cross correlation.

    Args:
        inp:  One-dimensional input array.

    Returns:
        Autocorrelation function for all positive lags.
    """
    N = len(inp)
    norm = inp @ inp
    out = np.empty(N)
    out[0] = 1
    for lag in range(1, N):
        pre = inp[:-lag]
        post = inp[lag:]
        prod = pre @ post
        if prod == 0:
            out[lag] = 0
        else:
            out[lag] = prod / norm
    return out


def acf_pearson(inp_sig):
    """Normalized estimate of the autocorrelation function of `inp_sig`
       by means of pearson correlation coefficient."""

    N = len(inp_sig)
    out = np.empty(N-1)
    out[0] = 1
    for lag in range(1, N-1):
        pre = inp_sig[:-lag]
        post = inp_sig[lag:]
        prod = corr_coef_pearson(pre, post)
        if prod == 0:
            out[lag] = 0
        else:
            out[lag] = prod
    return out


def corr_coef_pearson(x_sig: Array, y_sig: Array) -> float:
    """Fast perason correlation coefficient."""
    x_dtr = x_sig - np.mean(x_sig)
    y_dtr = y_sig - np.mean(y_sig)
    r_xy = np.convolve(x_dtr, y_dtr[::-1], mode='valid')
    r_xx_yy = (x_dtr @ x_dtr) * (y_dtr @ y_dtr)
    return r_xy / r_xx_yy


def c_weighting(frqs: Array) -> Array:
    """C-weighhting for SPL.

    Args:
        frq:    Frequencies.

    Returns:
        Weight for DFT bin with center frequency ``frq``.
    """
    aaa = 148693636.0
    bbb = 424.36
    sqf = np.power(frqs, 2)
    return np.divide(aaa*sqf, (sqf+aaa)*(sqf+bbb))


def freq2mel(frqs):
    """Transforms Hz to Mel-Frequencies.

    Params:
        frqs:  Frequency in Hz.

    Return:
        Frequency transformed to Mel scale.
    """
    frqs = np.atleast_1d(frqs)
    return 1125 * np.log(1 + frqs / 700)


def limit(inp: Array, ldb: Union[float] = None,
          udb: Union[float] = None) -> Array:
    """Limit the dynamic range of ``inp`` to  [``ldb``, ``udb``].

    Boundaries are given in dB SPL.

    Args:
        inp:  DFT bin magnitudes.
        ldb:  Lower clip boundary in deci Bel.
        udb:  Upper clip boundary in deci Bel.

    Returns:
        Copy of ``inp`` with values clipped.
    """
    if ldb is None:
        lth = 0.0
    elif isinstance(ldb, int) or isinstance(ldb, float):
        lth = amp(ldb)
    else:
        msg = (f'Argument to ``ldb`` must be of types ``int``, or ``float``.\n'
               f'Found {type(ldb)}.')
        raise TypeError(msg)

    if udb is None:
        uth = 0.0
    elif isinstance(udb, int) or isinstance(udb, float):
        uth = inp.max()
    else:
        msg = (f'Argument to ``udb`` must be of types ``int``, or ``float``.\n'
               f'Found {type(ldb)}.')
        raise TypeError(msg)

    low = np.where(inp < lth, 0.0, inp)
    return np.minimum(low, uth)


def mel2freq(zfrq):
    """Transforms Mel-Frequencies to Hzfrq.

    Args:
        zfrq:  Mel-Frequency.

    Returns:
        Frequency in Hz.
    """
    zfrq = np.atleast_1d(zfrq)
    return 700 * (np.exp(zfrq / 1125) - 1)


def maxamp(sig):
    """Maximal absolute elongation within the signal.

    Params:
        sig    (array-like) Input signal.

    Return:
        (scalar) Maximal amplitude.
    """
    return np.max(np.absolute(sig))


def minamp(sig):
    """Minimal absolute elongation within the signal.

    Params
        sig    (array-like) Input signal.

    Return:
        (scalar) Maximal amplitude.
    """
    return np.min(np.absolute(sig))


def noise(level, n=9000):
    """Generate withe noise.

    Params:
        level       (float) Noise level as standard deviation of a gaussian.
        n           (int) Length of noise signal in samples.

    Return:
        (ndarray)   White noise signal.
    """
    return stats.norm.rvs(0, level, size=n)


def normalize(sig):
    """Normlize a signal to [-1.0, 1.0].

    Params:
        sig (np.nadarray)    Signal to normalize.

    Return:
        (np.ndarray) Normalized signal.
    """
    return sig / np.max(np.absolute(sig), axis=0)


def sinusoid(frqs: Union[Sequence, Array, int, float],
             amps: Union[Sequence, Array, int, float] = 1,
             fps: int = 9000, length: float = 1.0,
             noise: float = None, comps: bool = False) -> Array:
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
        Array of signals.
    """
    frqs_: Array = np.atleast_1d(frqs)
    amps_: Array = np.atleast_1d(amps)

    if frqs_.shape == amps_.shape or amps_.size == 1:
        txs = np.arange(fps*length)[:, None] / fps
        sig = np.sin(2*np.pi*txs*frqs_) * amps_
    else:
        raise ValueError(f'Shape of ``frqs`` ({frqs_.shape}) differs from shape '
                         f' of ``amps``({amps_.shape}).')
    if noise:
        sig += stats.norm.rvs(0, noise, size=sig.shape)

    if comps:
        return sig
    return sig.sum(axis=1, keepdims=True)


def amp(spl: Union[Array, int, float],
        ref: float = _defaults.SPL_REF) -> Union[Array, float]:
    """Computes amplitudes form sound pressure level.

    The reference pressure defaults to the human hearing
    treshold of 20 μPa.

    Args:
        spl:    Sound pressure level.

    Returns:
        DFT magnituds.
    """
    return np.power(10.0, 0.05*spl) * ref


def zero_padding(sig: Array, n_pad: int,
                 dtype: Optional[Union[str, np.dtype]] = None) -> Array:
    """Append n zeros to signal. `sig` must be 1D array.

    Args:
        sig:    Array to be padded.
        n_pad:  Number of zeros to be appended.

    Returns:
        Zero-padded input signal.
    """
    if dtype is None:
        dtype = sig.dtype
    container = np.zeros(sig.size+n_pad, dtype=dtype)
    container[:sig.size] = sig
    return container
