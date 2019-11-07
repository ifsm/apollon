# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/signal/tools.py    (c) Michael Blaß 2016

Signal processing tools

Functions:
    acf                 Normalized autocorrelation.
    acf_pearson         Normalized Pearson acf.
    corr_coef_pearson   Correlation coefficient after Pearson.
    freq2mel            Transform frequency to mel.
    limit               Limit dynamic range.
    mel2freq            Transform mel to frequency.
    frq2bark            Transform frequency to Bark scale.
    maxamp              Maximal amplitude of signal.
    minamp              Minimal amplitude of signal.
    normalize           Scale data betwee -1.0 and 1.0.
    noise               Generate white noise.
    sinusoid            Generate sinusoidal signal.
    spl                 Conpute sound pressure level.
    zero_padding        Append array with zeros.
    trim_spectrogram    Trim spectrogram to a frequency range.
"""

import numpy as _np
import numpy.ma as _ma
from scipy import stats as _stats

from .. import _defaults
from .. types import Array as _Array
from .. import types as _types


def acf(inp_sig):
    """Normalized estimate of the autocorrelation function of `inp_sig`
       by means of cross correlation."""

    N = len(inp_sig)
    norm = inp_sig @ inp_sig

    out = _np.empty(N)
    out[0] = 1
    for m in range(1, N):
        a = inp_sig[:-m]
        b = inp_sig[m:]
        s = a @ b
        if s == 0:
            out[m] = 0
        else:
            out[m] = s / norm

    return out


def acf_pearson(inp_sig):
    """Normalized estimate of the autocorrelation function of `inp_sig`
       by means of pearson correlation coefficient."""

    N = len(inp_sig)
    out = _np.empty(N-1)

    out[0] = 1
    for m in range(1, N-1):

        a = inp_sig[:-m]
        b = inp_sig[m:]

        s = corr_coef_pearson(a, b)

        if s == 0:
            out[m] = 0
        else:
            out[m] = s

    return out


def trim_range(d_frq: float, lcf: float = None, ucf: float = None) -> slice:
    """Return slice of trim indices regarding an array ``frqs`` of DFT
    frquencies, such that both boundaries are included.

    Args:
        d_frq:   Frequency spacing.
        lcf:     Lower cut-off frequency.
        ucf:     Upper cut-off frequency.

    Returns:
        Slice of trim indices.
    """
    try:
        lcf = int(lcf//d_frq)
    except TypeError:
        lcf = None
    try:
        ucf = int(ucf//d_frq)
    except TypeError:
        ucf = None
    return slice(lcf, ucf)


def corr_coef_pearson(x, y):
    """Fast perason correlation coefficient."""
    detr_x = x - _np.mean(x)
    detr_y = y - _np.mean(y)

    r_xy = _np.convolve(detr_x, detr_y[::-1], mode='valid')
    r_xx_yy = (detr_x @ detr_x) * (detr_y @ detr_y)

    return r_xy / r_xx_yy


def freq2mel(f):
    """Transforms Hz to Mel-Frequencies.

    Params:
        f:    (real number) Frequency in Hz

    Return:
        (real number) Mel-Frequency
    """
    f = _np.atleast_1d(f)
    return 1125 * _np.log(1 + f / 700)


def limit(inp: _Array, ldb: float = None, udb: float = None):
    """Limit the dynamic range of ``inp`` to  [``ldb``, ``udb``].

    Boundaries are given in dB SPL.

    Args:
        inp:    DFT bin magnitudes.
        ldb:    Lower clip boundary in deci Bel.
        udb:    Upper clip boundary in deci Bel.

    Returns:
        Copy of ``inp`` with values clipped.
    """
    try:
        lth = amp(ldb)
    except TypeError:
        lth = 0.0

    try:
        uth = amp(udb)
    except TypeError:
        uth  = inp.max()

    low = _np.where(inp<lth, 0.0, inp)
    return _np.minimum(low, uth)


def mel2freq(z):
    """Transforms Mel-Frequencies to Hz.

    Params:
        z:     (real number) Mel-Frequency.

    Return:
        (real number) Frequency in Hz.
    """
    z = _np.atleast_1d(z)
    return 700 * (_np.exp(z / 1125) - 1)


def maxamp(sig):
    """Maximal absolute elongation within the signal.

    Params:
        sig    (array-like) Input signal.

    Return:
        (scalar) Maximal amplitude.
    """
    return _np.max(_np.absolute(sig))


def minamp(sig):
    """Minimal absolute elongation within the signal.

    Params
        sig    (array-like) Input signal.

    Return:
        (scalar) Maximal amplitude.
    """
    return _np.min(_np.absolute(sig))


def noise(level, n=9000):
    """Generate withe noise.

    Params:
        level       (float) Noise level as standard deviation of a gaussian.
        n           (int) Length of noise signal in samples.

    Return:
        (ndarray)   White noise signal.
    """
    return _stats.norm.rvs(0, level, size=n)


def normalize(sig):
    """Normlize a signal to [-1.0, 1.0].

    Params:
        sig (np.nadarray)    Signal to normalize.

    Return:
        (np.ndarray) Normalized signal.
    """
    return sig / _np.max(_np.absolute(sig), axis=0)


def sinusoid(frqs, amps=1, fps: int = 9000, length: float = 1.0,
             noise: float = None, comps: bool = False) -> _Array:
    """Generate sinusoidal signal.

    Args:
        frqs:    Component frequencies.
        amps:    Amplitude of each component in ``frqs``.  If ``amps`` is an
                 integer, each component of ``frqs`` is scaled according to
                 ``amps``. If ``amps` iis an iterable each frequency is scaled
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
    frqs = _np.atleast_1d(frqs)
    amps = _np.atleast_1d(amps)

    if frqs.shape == amps.shape or amps.size == 1:
        txs = _np.arange(fps*length)[:, None] / fps
        sig = _np.sin(2*_np.pi*txs*frqs) * amps
    else:
        raise ValueError(f'Shapes of ``frqs`` {frqs.shape} and ``amps``'
               '{ams.shape} differ.')

    if noise:
        sig += _stats.norm.rvs(0, noise, size=sig.shape)

    if comps:
        return sig
    return sig.sum(axis=1, keepdims=True)


def spl(amp: _Array, ref: float = _defaults.SPL_REF) -> _Array:
    """Computes sound pressure level.

    The values of ``amp`` are assumed to be magnitudes of DFT bins.

    The reference pressure defaults to the human hearing treshold of 20 μPa.

    This function sets all values of ``amp`` smaller then ``ref`` to ``ref``,
    hence eliminating inaudible singnal energy in the log domain.

    Args:
        amp:    Given amplitude values.

    Returns:
        Input scaled to deci Bel.
    """
    return 20.0 * _np.log10(_np.maximum(amp, ref) / ref)


def amp(spl: _Array, ref: float = _defaults.SPL_REF) -> _Array:
    """Computes amplitudes form sound pressure level.

    The reference pressure defaults to the human hearing treshold of 20 μPa.

    Args:
        spl:    Sound pressure level.

    Returns:
        DFT magnituds.
    """
    return _np.power(10.0, 0.05*spl) * ref


def zero_padding(sig: _Array, n_pad: int, dtype: str = None):
    """Append n zeros to signal. `sig` must be 1D array.

    Params:
        sig      Array to be padded.
        n_pad    Number of zeros to be appended.

    Return:
        Zero-padded input signal.
    """
    if dtype is None:
        dtype = sig.dtype
    container = _np.zeros(sig.size+n_pad, dtype=dtype)
    container[:sig.size] = sig
    return container


def clip_spectr(inp: _Array, frqs: _Array, lcf: float = None,
        ucf: float = None, dbt: float = None) -> _types.Spectrum:
    """Trim a spectral array to the frequency range [low, high].

    Additionally, clip amplitudes to ``dbt`` dB SPL.

    Args:
        inp:    Input spectrogram.
        frqs:   Spectrogram frequency axis.
        lcf:    Lower cut-off frequency.
        ucf:    Upper cut-off frequency.
        dbt:    Amplitude threshold in dB.

    Returns:
        Clipped spectral array, and frequencies.
    """
    lower_bound = frqs[0] if lcf is None else lcf
    upper_bound = frqs[-1] if ucf is None else ucf
    thr = _np.power(10, dbt/20) * _defaults.SPL_REF
    out_frqs = _ma.masked_outside(frqs, lower_bound, upper_bound)
    out_bins = _ma.masked_where(_np.absolute(inp) < thr, inp)

    return out_bins, out_frqs
