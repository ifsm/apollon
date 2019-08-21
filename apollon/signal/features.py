# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

"""
apollon/signal/features.py -- Feature extraction routines
"""

import numpy as _np
from scipy.signal import hilbert as _hilbert

import _features
from .. import segment as _segment
from .. tools import array2d_fsum
from .. types import Array as _Array
from .. import container
from .  import critical_bands as _cb
from .  tools import trim_spectrogram
from .. audio import fti16


def cdim(inp: _Array, delay: int, m_dim: int, n_bins: int = 1000,
         scaling_size: int = 10, mode: str = 'bader') -> _Array:
    # pylint: disable = too-many-arguments
    """Compute an estimate of the correlation dimension of the input data.

    If ``inp`` is two-dimensional, an estimated is computed for each row.

    Params:
        inp       Input array.
        delay     Embedding delay in samples.
        m_dim     Number of embedding dimensions.
        n_bins    Number of bins.
        mode      Use either 'bader' for the original algorithm
                  or 'blass' for the refined version.

    Returns:
        Array of correlation dimension estimates.

    Raises:
        ValueError
    """
    inp = _np.atleast_2d(inp)
    if inp.ndim < 1 or inp.ndim > 2:
        raise ValueError(f'Dimension of input array must not exceed 2. \
                Got {inp.ndim}')

    if mode == 'bader':
        cdim_func = _features.cdim_bader
        if inp.dtype != 'int16':
            inp = fti16(inp)
    elif mode == 'blass':
        raise NotImplementedError
        #cdim_func = fractal.cdim
    else:
        raise ValueError(f'Unknown mode "{mode}". Expected either "bader", \
                or "blass"')

    return _np.array([cdim_func(seg, delay, m_dim, n_bins, scaling_size)
                      for seg in inp])


def spectral_centroid(inp: _Array, frqs: _Array) -> _Array:
    """Estimate the spectral centroid frequency.

    Calculation is applied to the second axis. One-dimensional
    arrays will be promoted.

    Args:
        inp:   Input data. Each row is assumend FFT bins scaled by `freqs`.
        frqs:  One-dimensional array of FFT frequencies.

    Returns:
        Array of Spectral centroid frequencies.
    """
    inp = _np.atleast_2d(inp).astype('float64')

    weighted_nrgy = _np.multiply(inp, frqs).sum(axis=1).squeeze()
    total_nrgy = inp.sum(axis=1).squeeze()
    total_nrgy[total_nrgy == 0.0] = 1.0

    return _np.divide(weighted_nrgy, total_nrgy)


def spectral_flux(inp: _Array, delta: float = 1.0) -> _Array:
    """Estimate the spectral flux

    Args:
        inp:    Input data. Each row is assumend FFT bins.
        delta:  Sample spacing.

    Returns:
        Array of Spectral flux.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    return _np.maximum(_np.gradient(inp, delta, axis=-1), 0).squeeze()


def spectral_shape(inp: _Array, frqs: _Array, cf_low: float = 50,
                   cf_high: float = 16000) -> container.FeatureSpace:
    """Compute low-level spectral shape descriptors.

    This function computes the first four central moments of
    the input spectrum. If input is two-dimensional, the first
    axis is assumed to represent frequency.

    The central moments are:
        - Spectral Centroid (SC)
        - Spectral Spread (SSP),
        - Spectral Skewness (SSK)
        - Spectral Kurtosis (SKU).

    Spectral Centroid represents the center of gravity of the spectrum.
    It correlates well with the perception of auditory brightness.

    Spectral Spread is a measure for the frequency deviation around the
    centroid.

    Spectral Skewness is a measure of spectral symmetry. For values of
    SSK = 0 the spectral distribution is exactly symmetric. SSK > 0 indicates
    more power in the frequency domain below the centroid and vice versa.

    Spectral Kurtosis is a measure of flatness. The lower the value, the faltter
    the distribution.

    Args:
        inp:      Input spectrum or spectrogram.
        frqs:     Frequency axis.
        cf_low:   Lower cutoff frequency.
        cf_high:  Upper cutoff frequency.

    Returns:
        Spectral centroid, spread, skewness, and kurtosis.
    """
    if inp.ndim < 2:
        inp = inp[:, None]

    vals, frqs = trim_spectrogram(inp, frqs, cf_low, cf_high)

    total_nrgy = array2d_fsum(vals)
    total_nrgy[total_nrgy == 0.0] = 1.0    # Total energy is zero iff input signal is all zero.
                                           # Replace these bin values with 1, so division by
                                           # total energy will not lead to nans.

    centroid = frqs @ vals / total_nrgy
    deviation = frqs[:, None] - centroid

    spread = array2d_fsum(_np.power(deviation, 2) * vals)
    skew = array2d_fsum(_np.power(deviation, 3) * vals)
    kurt = array2d_fsum(_np.power(deviation, 4) * vals)

    spread = _np.sqrt(spread/total_nrgy)
    zero_spread = spread == 0

    skew = _np.divide(skew/total_nrgy, _np.power(spread, 3), where=~zero_spread)
    kurt = _np.divide(kurt/total_nrgy, _np.power(spread, 4), where=~zero_spread)
    skew[zero_spread] = 0
    kurt[zero_spread] = 0

    return container.FeatureSpace(centroid=centroid, spread=spread, skewness=skew, kurtosis=kurt)


def log_attack_time(inp: _Array, fps: int, ons_idx: _Array,
                    wlen: float = 0.05) -> _Array:
    """Estimate the attack time of each onset and return its logarithm.

    This function estimates the attack time as the duration between the
    onset and the local maxima of the magnitude of the Hilbert transform
    of the local window.

    Args:
        inp:      Input signal.
        fps:      Sampling frequency.
        ons_idx:  Sample indices of onsets.
        wlen:     Local window length in samples.

    Returns:
        Logarithm of the attack time.
    """
    wlen = int(fps * wlen)
    segs = _segment.by_onsets(inp, wlen, ons_idx)
    attack_time = _np.absolute(_hilbert(segs)).argmax(axis=1) / fps
    attack_time[attack_time == 0.0] = 1.0
    return _np.log(attack_time)


def perceptual_shape(inp: _Array, frqs: _Array, cf_low: float = 50,
                     cf_high: float = 16000) -> container.FeatureSpace:
    """Extracts psychoacoustical features from the spectrum.

    Returns:
        Loudness, roughness, and sharpness.
    """
    if inp.ndim < 2:
        inp = inp[:, None]

    cbrs = _cb.filter_bank(frqs) @ inp
    loud_specific = _np.maximum(_cb.specific_loudness(cbrs),
                                _np.finfo('float64').eps)
    loud_total = array2d_fsum(loud_specific, axis=0)


    zfn = _np.arange(1, 25)
    sharp = ((zfn * _cb.weight_factor(zfn)) @ cbrs) / loud_total
    rough = hrough(inp, frqs)

    return container.FeatureSpace(loudness=loud_total, sharpness=sharp, roughness=rough)


def loudness(inp: _Array, frqs: _Array) -> _Array:
    """Calculate a measure for the perceived loudness from a spectrogram.

    Args:
        inp:  Magnitude spectrogram.

    Returns:
        Loudness
    """
    cbrs = _cb.filter_bank(frqs) @ inp
    return _cb.total_loudness(cbrs)


def hrough(bins: _Array, frqs: float, cf_low: float = 50,
           cf_high: float = 16000, min_intensity: float = -48) -> _Array:
    """Calculate Helmholtz Roughness from spectrogram.

    Helmholtz Roughness assumes that the maximal roughness
    occures at frequency distance of 33 Hz independently of
    the frequency und consideration. Hence, the input spctrogram
    should have sufficiently high frequency resolution.


    Args:
        bins:   Input spectral magnitudes.
        frqs:   Frequency axis of spectrogram.
        cf_low: Lower frequency bound.
        cf_high: Upper requenc bound.
        min_intensity:  Intensity threshold in dB.

    Returns:
        Roughness per spectrogram time instant.
    """
    d_frq = frqs[1] -  frqs[0]
    max_bin = int(round(50 // d_frq))

    frange = frqs[1:max_bin] / (33.5 * _np.exp(-1))
    fdecay = _np.exp(-frqs[1:max_bin] / 33.5)
    r_curve = frange * fdecay

    logspc = 20 * _np.log10(bins/bins.max())
    bins[logspc < min_intensity] = 0

    rr = _np.zeros(bins.shape[1])

    if cf_high + max_bin < bins.shape[0]:
        upper_limit = cf_high
    else:
        upper_limit = bins.shape[0]

    upper_limit = cf_high if cf_high + max_bin < bins.shape[0] else bins.shape[0]
    for i in range(cf_low, upper_limit, max_bin):
        amps = bins[i] * bins[i+1:i+max_bin]
        rr += _np.sum(amps * r_curve.reshape(-1, 1), axis=0)
    return rr


def sharpness(inp: _Array, frqs: _Array) -> _Array:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram.

    Args:
        inp:   Two-dimensional input array. Assumed to be an magnitude spectrogram.
        frqs:  Frequency axis of the spectrogram.

    Returns:
        Sharpness for each time instant of the spectragram.
    """
    cbrs = _cb.filter_bank(frqs) @ inp
    return _cb.sharpness(cbrs)
