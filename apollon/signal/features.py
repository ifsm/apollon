# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
apollon/signal/features.py -- Feature extraction routines

Functions:
    cdim           Fractal correlation dimension.
    correlogram    Windowed auto-correlation.
    energy    Total signal energy.
    rms    Root mean square.
    spectral_centroid
    spectral_spread
    spectral_flux
    spl
    splc
    loudness
    sharpness
    roughness
"""

import numpy as _np
from scipy.signal import hilbert as _hilbert

import _features
from . import tools as _sigtools
from .. import segment as _segment
from .. import tools
from .. types import Array as _Array
from .. import container
from .  import critical_bands as _cb
from .. audio import fti16
from .. import _defaults


def cdim(inp: _Array, delay: int, m_dim: int, n_bins: int = 1000,
         scaling_size: int = 10, mode: str = 'bader') -> _Array:
    # pylint: disable = too-many-arguments
    """Compute an estimate of the correlation dimension of the input data.

    If ``inp`` is two-dimensional, an estimated is computed for each row.

    Args:
        inp:       Input array.
        delay:     Embedding delay in samples.
        m_dim:     Number of embedding dimensions.
        n_bins:    Number of bins.
        mode:      Use either 'bader' for the original algorithm
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

    cdim = _np.array([cdim_func(seg, delay, m_dim, n_bins, scaling_size)
            for seg in inp])
    return _np.nan_to_num(cdim, posinf=0, neginf=0)

def correlogram(inp: _Array, wlen: int, n_delay: int,
        total: bool = False) -> _Array:
    """Windowed autocorrelation of ``inp``.

    This function computes the autocorrelation of a ``wlen``-sized
    window of the input.

    Args:
        inp:        One-dimensional input signal.
        wlen:       Length of the autocorrelation window.
        n_delay:    Number of delay.

    Returns:
        Two-dimensional array in which each column is an auto-correlation
        function.
    """
    if not isinstance(inp, _np.ndarray):
        raise TypeError(f'Argument ``inp`` is of type {type(inp)}. It has '
                        'to be an numpy array.')

    crr = _features.correlogram(inp, wlen, n_delay)
    if total is True:
        return crr.sum(keepdims=True) / _np.prod(crr.shape)
    return crr


def energy(sig: _Array) -> _Array:
    """Total energy of time domain signal.

    Args:
        sig:  Time domain signal.

    Returns:
        Energy along fist axis.
    """
    return np.sum(np.square(np.abs(sig)), axis=0)


def frms(bins: _Array, n_sig: int, window: str = None) -> _Array:
    """Root meann square of signal energy estimate in the specrtal domain.

    Args:
        bins:    DFT bins.
        n_sig:   Size of original signal.
        window:  Window function applied to original signal.

    Returns:
        Estimate of signal energy along first axis.
    """
    vals = bins * n_sig
    if n_sig % 2:
        vals /= 2
    else:
        vals[:-1] /= 2
    foo = np.sqrt(2*energy(vals)) / n_sig
    if window:
        foo /= rms(getattr(np, window)(n_sig))
    return foo


def rms(sig: _Array) -> _Array:
    """Root mean square of time domain signal.

    Args:
        sig:  Time domain signal

    Returns:
        RMS of signal along first axis.
    """
    return np.sqrt(np.mean(np.square(np.abs(sig)), axis=0))


def spectral_centroid(frqs: _Array, bins: _Array) -> _Array:
    """Estimate the spectral centroid frequency.

    Spectral centroid is always computed along the second axis of ``bins``.

    Args:
        frqs:     Nx1 array of DFT frequencies.
        power:    NxM array of DFT bin values.

    Returns:
        1xM array of spectral centroids.
    """
    return tools.fsum(frqs*_power_distr(bins), axis=0, keepdims=True)


def spectral_flux(inp: _Array, delta: float = 1.0) -> _Array:
    """Estimate the spectral flux

    Args:
        inp:      Input data. Each row is assumend FFT bins.
        delta:    Sample spacing.

    Returns:
        Array of Spectral flux.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    return _np.maximum(_np.gradient(inp, delta, axis=-1), 0).squeeze()


def spectral_spread(frqs: _Array, bins: _Array) -> _Array:
    """Estimate spectral spread.

    Spectral spread is always computed along the second axis of ``bins``.
    This function computes the square roote of spectral spread.

    Args:
        frqs:     Nx1 array of DFT frequencies.
        power:    NxM array of DFT bin values.

    Returns:
        Square root of spectral spread.
    """
    deviation = _np.power(frqs-spectral_centroid(frqs, bins), 2)
    return _np.sqrt(tools.fsum(deviation*_power_distr(bins), axis=0,
        keepdims=True))


def spl(amps: _Array, total: bool = False, ref: float = None) -> _Array:
    """Computes sound pressure level.

    The values of ``amp`` are assumed to be magnitudes of DFT bins.

    The reference pressure defaults to the human hearing treshold of 20 μPa.

    Args:
        amps:     Amplitude values.
        total:    If True, returns the total spl over all values. In case
                  ``amp`` is two-dimensional, the first axis is aggregated.
        ref:      Custom reference value.

    Returns:
        Sound pressure level of ``amp``.
    """
    if ref is None:
        ref = _defaults.SPL_REF

    vals = _np.power(amps/ref, 2)
    if total:
        vals = vals.sum(axis=0, keepdims=True)
    vals = _np.maximum(1.0, vals)
    return 10.0*_np.log10(vals)


def splc(frqs: _Array, amps: _Array, total: bool = False,
         ref: float = None) -> _Array:
    """Apply C-weighted to SPL.

    Args:
        frqs:    Center frequency of DFT band.
        amps:    Magnitude of DFT band.
        ref:     Reference value for p_0.

    Returns:
        C-weighted sound pressure level.
    """
    return spl(_sigtools.c_weighting(frqs)*amps, total)


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


def loudness(frqs: _Array, bins: _Array) -> _Array:
    """Calculate a measure for the perceived loudness from a spectrogram.

    Args:
        frqs:   Frquency axis.
        bins:   Magnitude spectrogram.

    Returns:
        Estimate of the total loudness.
    """
    cbrs = _cb.filter_bank(frqs) @ bins
    return _cb.total_loudness(cbrs)


def roughness_helmholtz(frqs: _Array, bins: _Array, frq_max: float,
        total: bool = True) -> _Array:
    frq_res = (frqs[1]-frqs[0]).item()
    kernel = _roughnes_kernel(frq_res, frq_max)
    out = _np.correlate(bins.squeeze(), kernel, mode='same')[:, None]
    if total is True:
        out = out.sum(keepdims=True)
    return out


def sharpness(frqs: _Array, bins: _Array) -> _Array:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram.

    Args:
        frqs:    Frequencies.
        bins:    DFT magnitudes.

    Returns:
        Sharpness.
    """
    cbrs = _cb.filter_bank(frqs.squeeze()) @ bins.squeeze()
    return _np.expand_dims(_cb.sharpness(cbrs), 1)


def _power_distr(bins: _Array) -> _Array:
    """Computes the spectral energy distribution.

    Args:
        bins:    NxM array of DFT bins.

    Returns:
        NxM array of spectral densities.
    """
    total_power = tools.fsum(bins, axis=0, keepdims=True)
    total_power[total_power==0] = 1
    return bins / total_power


def _roughnes_kernel(frq_res: float, frq_max: float) -> _Array:
    """Comput the convolution kernel for roughness computation.

    Args:
        frq_res:    Frequency resolution
        frq_max:    Frequency bound.

    Returns:
        Weight for each frequency below ``frq_max``.
    """
    frm = 33.5
    bin_idx = int(_np.round(frq_max/frq_res))
    norm = frm * _np.exp(-1)
    base = _np.abs(_np.arange(-bin_idx, bin_idx+1)) * frq_res
    return base / norm * _np.exp(-base/frm)

