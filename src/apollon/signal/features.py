"""
Feature extraction routines
============================
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as _np
from scipy.signal import hilbert as _hilbert
from scipy.signal import correlate, find_peaks

from . import _features     # pylint: disable = no-name-in-module
from . import tools as _sigtools
from .. import segment as _segment
from .. typing import Array, FloatArray, floatarray, IntArray
from . import critical_bands as _cb
from .. import _defaults

if TYPE_CHECKING:
    from . spectral import TransformResult


def cdim(inp: FloatArray, delay: int, m_dim: int, n_bins: int = 1000,
         scaling_size: int = 10, mode: str = 'bader') -> FloatArray:
    # pylint: disable = too-many-arguments
    r"""Compute an estimate of the correlation dimension ``inp``.

    This function implements the Grassberger-Procaccia algorithm
    [Grassberger1983]_ to compute the correlation sum

    .. math::
        \hat C(r) = \frac{2} {N(n-1)} \sum_{i<j}
        \Theta (r - | \boldsymbol{x}_i - \boldsymbol{x}_j)

    from a time delay embedding of ``inp``.

    If ``mode`` is set to 'bader', each column of ``inp`` must hold at least
    ``2390 + (m_dim-1)*delay`` samples, and ``scaling_size`` must not exceed
    ``n_bins - floor(0.6*n_bins)``, i.e., 400 for the default ``n_bins``.

    Args:
        inp:       Input array.
        delay:     Embedding delay in samples.
        m_dim:     Number of embedding dimensions.
        n_bins:    Number of bins.
        scaling_size:  Distance in bins between the points of the slope.
        mode:      Use either 'bader' for the original algorithm

    Returns:
        Array of correlation dimension estimates.

    Raises:
        ValueError: If ``inp`` is not two-dimensional, if ``mode`` is
            unknown, if ``delay``, ``m_dim``, ``n_bins``, or ``scaling_size``
            is not positive, or if ``inp`` or ``n_bins`` is too small for
            the requested embedding and ``scaling_size``.

    .. [Grassberger1983] P. Grassberger, and I. Procaccia,
       "Measuring the strangeness of strange attractors,"  *Physica 9d*, pp. 189-208.
    """
    if inp.ndim != 2:
        raise ValueError('Input array must be two-dimensional.')

    if mode == 'bader':
        cdim_func = _features.cdim_bader
        inp_: Array
        if inp.dtype == 'int16':
            inp_ = inp.copy()
        else:
            inp_ = _sigtools.fti16(inp)

    elif mode == 'blass':
        raise NotImplementedError
        # cdim_func = fractal.cdim
    else:
        raise ValueError(f'Unknown mode "{mode}". Expected either "bader", '
                         'or "blass"')
    out = _np.zeros(inp_.shape[1])
    for i, seg in enumerate(inp_.T):
        out[i] = _np.nan_to_num(cdim_func(seg, delay, m_dim, n_bins,
                                          scaling_size))
    return _np.expand_dims(out, 0)


def correlogram(inp: FloatArray, wlen: int, n_delay: int,
                total: bool = False) -> FloatArray:
    r"""Windowed autocorrelation of ``inp``.

    This function estimates autocorrelation functions between ``wlen``-sized
    windows of the input, separated by ``n_delay`` samples [Granqvist2003]_ .
    The autocorrelation :math:`r_{m, n}` is given by

    .. math::
        r_{m, n} = \frac{ \sum_{k=m}^{m+w-1} (x_k- \overline x_m)(x_{k+m}-
        \overline x_{m+n})}
        {\sqrt{\sum_{k=m}^{m+w-1}(x_k - \overline x_m)^2
        \sum_{k=m}^{m+w-1}(x_{k+n} - \overline x_{m+n})^2}} \,,

    where :math:`x_m` is

    .. math::
        x_m=\frac{\sum_{i=m}^{m+w-1} x_i}{w} \,.

    Args:
        inp:        One-dimensional input signal.
        wlen:       Length of the autocorrelation window.
        n_delay:    Number of delay.
        total:      Sum the correlogram along its first axis.

    Returns:
        Two-dimensional array in which each column is an auto-correlation
        function.

    .. [Granqvist2003] S. Granqvist, B. Hammarberg, 
       "The correlogram: a visual display of periodicity," *JASA,* 114, pp. 2934.
    """
    if not isinstance(inp, _np.ndarray):
        raise TypeError(f'Argument ``inp`` is of type {type(inp)}. It has '
                        'to be an numpy array.')

    if inp.ndim != 2:
        raise ValueError('Input must be two-dimensional.')

    out = _np.zeros((inp.shape[1], n_delay, inp.shape[0]-wlen-n_delay), dtype=_np.double)
    for i, seg in enumerate(inp.T):
        out[i] = _features.correlogram(seg, wlen, n_delay)
    if total is True:
        return floatarray(out.sum(axis=(1, 2)) / _np.prod(out.shape[1:]))
    return out


def energy(sig: FloatArray) -> FloatArray:
    """Total energy of time domain signal.

    Args:
        sig:  Time domain signal.

    Returns:
        Energy along fist axis.
    """
    if not _np.isfinite(sig).all():
        raise ValueError('Input ``sig`` contains NaNs or infinite values.')
    buff = _np.empty_like(sig, dtype=_np.double)
    _np.abs(sig, out=buff)
    _np.square(buff, out=buff)
    total = _np.empty((1, buff.shape[1]))
    return _np.sum(buff, axis=0, dtype=_np.double, out=total, keepdims=True)


def rms(sig: FloatArray) -> FloatArray:
    """Root mean square of time domain signal.

    Args:
        sig:  Time domain signal

    Returns:
        RMS of signal along first axis.
    """
    buff = _np.empty_like(sig, dtype=_np.double)
    out = _np.empty((1, sig.shape[1]), dtype=_np.double)
    _np.abs(sig, out=buff)
    _np.square(buff, out=buff)
    _np.mean(buff, axis=0, keepdims=True, out=out)
    _np.sqrt(out, out=out)
    return out


def spectral_centroid(frqs: FloatArray, amps: FloatArray) -> FloatArray:
    r"""Estimate the spectral centroid frequency.

    Spectral centroid is always computed along the second axis of ``amps``.

    Args:
        frqs:   Nx1 array of DFT frequencies.
        amps:   NxM array of absolute values of DFT bins.

    Returns:
        1xM array of spectral centroids.

    Note:
        The spectral centroid frequency :math:`f_C` is computed as
        the expectation of a spectral distribution:

        .. math::
            f_C = \sum_{i=0}^{N} f_i p(i) \,,

        where :math:`f_i` is the center frequency, and :math:`p(i)` the
        relative amplitude of the :math:`i` th DFT bin.
    """
    out = _np.empty((1, amps.shape[1]), dtype=_np.double)
    return _np.sum(frqs*_power_distr(amps), axis=0, keepdims=True, out=out)


def spectral_spread(frqs: FloatArray, bins: FloatArray,
                    centroids: FloatArray | None = None) -> FloatArray:
    r"""Estimate spectral spread.

    Spectral Spread is always computed along the second axis of ``bins``.
    This function computes the square roote of spectral spread.

    Args:
        frqs:   Nx1 array of DFT frequencies.
        bins:   NxM array of DFT bin values.
        centroids:  Array Spectral Centroid values.

    Returns:
        Square root of spectral spread.

    Note:
        Spectral Spread :math:`f_s` is computed as

        .. math::
            f_S = \sum_{i=0}^N (f_i - f_C)^2 p(i) \,,

        where :math:`f_i` is the center frequency, and :math:`p(i)` the
        relative amplitude of the :math:`i` th DFT bin. :math:`f_C` is the
        spectral centroid frequency.
    """
    if centroids is None:
        centroids = spectral_centroid(frqs, bins)
    deviation = _np.power(frqs-centroids, 2)
    return floatarray(_np.sqrt(_np.sum(deviation*_power_distr(bins), axis=0,
                            keepdims=True)))


def spectral_skewness(frqs: FloatArray, bins: FloatArray,
                      centroid: FloatArray | None = None,
                      spreads: FloatArray | None = None) -> FloatArray:
    r"""Estimate the spectral skewness.

    Args:
        frqs:   Frequency array.
        bins:   Absolute values of DFT bins.
        centroids:  Precomputed spectral centroids.
        spreads:    Precomputed spectral spreads.

    Returns:
        Array of spectral skewness values.

    Note:
        The spectral skewness :math:`S_S` is calculated by

        .. math::
            S_{K} = \sum_{i=0}^N \frac{(f_i-f_C)^3}{\sigma^3} p(i) \,,

        where :math:`f_i` is the center frequency, and :math:`p(i)` the
        relative amplitude of the :math:`i` th DFT bin. :math:`f_C` is the
        spectral centroid frequency, and :math:`\sigma = \sqrt{f_S}.`
    """
    raise NotImplementedError

def spectral_kurtosis(frqs: FloatArray, bins: FloatArray,
                      centroid: FloatArray | None = None,
                      spreads: FloatArray | None = None) -> FloatArray:
    r"""Estimate spectral kurtosis.

    Args:
        frqs:   Frequency array.
        bins:   Absolute values of DFT bins.
        centroids:  Precomputed spectral centroids.
        spreads:    Precomputed spectral spreads.

    Returns:
        Array of spectral skewness values.

    Note:
        Spectral kurtosis is calculated by

        .. math::
            S_{K} = \sum_{i=0}^N \frac{(f_i-f_c)^4}{\sigma^4} p(i) \,,

        where :math:`f_i` is the center frequency, and :math:`p(i)` the
        relative amplitude of the :math:`i` th DFT bin. :math:`f_C` is the
        spectral centroid frequency, and :math:`\sigma = \sqrt{f_S}.`
    """
    raise NotImplementedError

def spectral_flux(inp: FloatArray, delta: float = 1.0,
                  total: bool = True) -> FloatArray:
    r"""Estimate the spectral flux

    The flux of a frame is the rectified increase of each bin over the
    preceding frame. The first frame has no predecessor and reads 0.

    Args:
        inp:    Magnitude spectrogram, shaped ``(n_frqs, n_frames)``. Each
                column is a spectrum.
        delta:  Spacing of the frames. The differences are divided by it.
        total:  If ``True``, sum over the frequency axis.

    Returns:
        Spectral flux per frame, shaped ``(1, n_frames)``, or per bin and
        frame if ``total`` is ``False``.

    Note:
        Spectral flux is computed by

        .. math::
            SF(i) = \sum_{j=0}^k H(|X_{i,j}| - |X_{i-1,j}|) \,,

        where :math:`X_{i,j}` is the :math:`j` th frequency bin of the :math:`i`
        th spectrum :math:`X` of a spectrogram :math:`\boldsymbol X`.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    out = _np.diff(inp, axis=-1, prepend=inp[..., :1]) / delta
    _np.maximum(out, 0, out=out)
    if total:
        return floatarray(out.sum(axis=0, keepdims=True))
    return out


def spl(inp: FloatArray, ref: float = _defaults.SPL_REF) -> FloatArray:
    """Computes the average sound pressure level of time domain signal.

    Args:
        inp:  Time domain signal.
        ref:  Reference level.

    Returns:
        Average sound pressure level. Silent channels read ``-inf``.
    """
    ratio = rms(inp)/ref
    level = _np.full_like(ratio, -_np.inf)
    _np.log10(ratio, where=ratio>0, out=level)
    _np.multiply(level, 20.0, out=level)
    return level


def log_attack_time(inp: FloatArray, fps: int, ons_idx: IntArray,
                    wlen: float = 0.05) -> FloatArray:
    """Estimate the attack time of each onset and return its logarithm.

    This function estimates the attack time as the duration between the
    onset and the maximum of the magnitude of the Hilbert transform of the
    window following it. An envelope that peaks at the onset itself has an
    attack shorter than the sampling resolution; its attack time is floored
    at one sample, ``1/fps``, so that the fastest attacks read the smallest
    values.

    Args:
        inp:      One-dimensional input signal.
        fps:      Sampling frequency.
        ons_idx:  Sample indices of onsets.
        wlen:     Length of the window following each onset in seconds.

    Returns:
        Natural logarithm of the attack time in seconds, one per onset.

    Raises:
        ValueError: If ``inp`` is not one-dimensional.
    """
    if inp.ndim != 1:
        raise ValueError(f'``inp`` has {inp.ndim} dimensions. Expected a '
                         'one-dimensional signal.')
    n_wlen = int(fps * wlen)
    segs = _segment.by_onsets(inp, n_wlen, ons_idx)
    n_attack = _np.absolute(_hilbert(segs)).argmax(axis=1)
    return floatarray(_np.log(_np.maximum(n_attack, 1) / fps))


def loudness(sxx: TransformResult, resolution: float = 0.1) -> FloatArray:
    """Calculate a measure for the perceived loudness from a spectrogram.

    The model works on absolute sound pressure levels, so the signal behind
    ``sxx`` must be calibrated in Pa. Its power is taken from
    ``sxx.ms_power``, which holds whatever scaling ``sxx`` was computed with.

    Args:
        sxx:    Spectrum or spectrogram of a signal in Pa.
        resolution: Bark width of the fine excitation-spreading grid (see
            ``critical_bands.excitation_pattern``); smaller is more
            accurate but more expensive.

    Returns:
        Estimate of the total loudness.
    """
    cbrs = _cb.excitation_pattern(sxx.frqs.squeeze(), sxx.ms_power, resolution)
    return _cb.total_loudness(cbrs, spread_input=False)


def roughness_helmholtz(d_frq: float, bins: FloatArray, frq_max: float,
                        total: bool = True) -> FloatArray:
    """Estimate a relative roughness index using Helmholtz' algorithm.

    Each spectrum is reduced to its partials, its local maxima relative to
    the largest bin. Partials fade in between 5 % and 15 % of the largest
    bin, so that window sidelobes and noise count little, and no partial
    appears suddenly. Their autocorrelation holds, for each lag k, how
    strongly pairs of partials ``k*d_frq`` Hz apart are present. Each lag is
    weighted by Helmholtz' roughness curve, which peaks at a spacing of
    33.5 Hz, and taken relative to the power of the partials, i.e., to the
    autocorrelation at lag 0. Two equal partials hence read exactly the
    value of the curve at their spacing, and the index changes continuously
    with the amplitudes of the partials.

    The measure is relative and independent of level. Partials closer than
    the main lobe of the window merge into one. With a Hann window that is
    about three bins, so a 33 Hz spacing needs ``d_frq`` of about 11 Hz or
    less, i.e., 4096 samples per segment at 44.1 kHz. Use a window with low
    sidelobes, such as Hann: the sidelobes of a rectangular window reach
    22 % of the main lobe, and read as partials whenever a partial falls
    between bins.

    The result is a dimensionless index, not a roughness in asper, and it
    does not follow the psychoacoustic data behind that unit:

    - It ignores level, whereas perceived roughness grows with it.
    - Its curve peaks at a spacing of 33.5 Hz, whereas the roughness of a
      1 kHz tone peaks at a modulation frequency of about 70 Hz. The asper
      reference stimulus, that tone fully modulated at 70 Hz and 60 dB SPL,
      reads about 1.0 only by coincidence: modulated at 30 Hz, it reads 1.6.
    - It grows with modulation depth, but not with the roughly 1.6th power
      of the depth that perceived roughness follows.

    Compare spectra with it only when they were analysed alike.

    Args:
        d_frq:      Frequency resolution of ``bins`` in Hz.
        bins:       Magnitude spectrogram, shaped ``(n_frqs, n_frames)``.
        frq_max:    Highest frequency considered in Hz. Partials above it are
                    ignored, and spacings up to it are weighted.
        total:      If ``True``, sum the contributions of all spacings.

    Returns:
        Roughness index per frame, shaped ``(1, n_frames)``. If ``total`` is
        ``False``, the contribution of each spacing ``0, d_frq, ..., frq_max``
        instead, shaped ``(n_spacings, n_frames)``.

    Raises:
        ValueError: If ``bins`` is not two-dimensional, or if ``frq_max``
            exceeds the highest frequency of ``bins``.
    """
    if bins.ndim != 2:
        raise ValueError(f'``bins`` has {bins.ndim} dimensions. Expected a '
                         'spectrogram shaped (n_frqs, n_frames).')

    kernel = _roughness_kernel(d_frq, frq_max)
    if kernel.size > bins.shape[0]:
        raise ValueError(f'``frq_max`` ({frq_max} Hz) exceeds the highest '
                         f'frequency of ``bins`` ({(bins.shape[0]-1)*d_frq} Hz).')

    out = _np.zeros((kernel.size, bins.shape[1]))
    for i, frame in enumerate(bins[:kernel.size].T):
        partials = _partials(frame)
        acr = correlate(partials, partials)[partials.size-1:]
        if acr[0] > 0:
            # Each pair appears twice in the full autocorrelation, and lag 0
            # carries no weight, since the curve vanishes there.
            out[:, i] = 2 * acr * kernel / acr[0]

    if total:
        out = out.sum(axis=0, keepdims=True)
    return out


def sharpness(sxx: TransformResult, resolution: float = 0.1) -> FloatArray:
    """Calculate a measure for the perception of auditory sharpness from a
    spectrogram.

    Masking spreads further at higher levels, so sharpness depends on the
    absolute sound pressure level, too. The signal behind ``sxx`` must hence
    be calibrated in Pa. Its power is taken from ``sxx.ms_power``, which
    holds whatever scaling ``sxx`` was computed with.

    Args:
        sxx:    Spectrum or spectrogram of a signal in Pa.
        resolution: Bark width of the fine excitation-spreading grid (see
            ``critical_bands.excitation_pattern``); smaller is more
            accurate but more expensive.

    Returns:
        Sharpness.
    """
    cbrs = _cb.excitation_pattern(sxx.frqs.squeeze(), sxx.ms_power, resolution)
    return _cb.sharpness(cbrs, spread_input=False)


def _power_distr(bins: FloatArray) -> FloatArray:
    """Computes the spectral energy distribution.

    Args:
        bins:    NxM array of DFT bins.

    Returns:
        NxM array of spectral densities.
    """
    total_power = _np.empty((1, bins.shape[1]), dtype=_np.double)
    _np.sum(bins, axis=0, keepdims=True, out=total_power)
    total_power[total_power == 0] = 1
    return bins / total_power


def _partials(spectrum: FloatArray) -> FloatArray:
    """Reduce a magnitude spectrum to its partials.

    Each local maximum is divided by the largest bin and faded in by that
    ratio: it is dropped up to 5 %, kept fully from 15 %, and scaled
    linearly in between.

    Args:
        spectrum:  One-dimensional magnitude spectrum.

    Returns:
        New array holding the faded partials, and zero elsewhere.
    """
    fade_start, fade_stop = 0.05, 0.15
    out = _np.zeros(spectrum.shape, dtype=_np.double)
    peak = spectrum.max()
    if peak > 0:
        idx, _ = find_peaks(spectrum)
        rel = spectrum[idx] / peak
        fade = _np.clip((rel-fade_start) / (fade_stop-fade_start), 0.0, 1.0)
        out[idx] = rel * fade
    return out


def _roughness_kernel(frq_res: float, frq_max: float) -> FloatArray:
    """Compute Helmholtz' roughness curve for the spacings of partials.

    The curve ``g(f) = f/f_m * exp(1 - f/f_m)`` with ``f_m = 33.5`` Hz peaks
    at 1 for a spacing of ``f_m``, and vanishes for coinciding partials.

    Args:
        frq_res:    Frequency resolution in Hz.
        frq_max:    Largest spacing in Hz.

    Returns:
        Weight for each spacing ``0, frq_res, ..., frq_max``.
    """
    frm = 33.5
    base = _np.arange(int(round(frq_max/frq_res)) + 1) * frq_res
    return floatarray(base / frm * _np.exp(1 - base/frm))
