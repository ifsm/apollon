"""
Audio feature extraction routines
"""

import numpy as _np
from scipy.signal import hilbert as _hilbert

from . import _features     # pylint: disable = no-name-in-module
from . import tools as _sigtools
from .. import segment as _segment
from .. types import FloatArray, floatarray, IntArray, NDArray
from . import critical_bands as _cb
from .. audio import fti16
from .. import _defaults


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

    If ``mode`` is set to 'bader', the input array must have at least
    2400 elements. Otherwise, the result is undefined.

    Args:
        inp:       Input array.
        delay:     Embedding delay in samples.
        m_dim:     Number of embedding dimensions.
        n_bins:    Number of bins.
        mode:      Use either 'bader' for the original algorithm

    Returns:
        Array of correlation dimension estimates.

    Raises:
        ValueError

    .. [Grassberger1983] P. Grassberger, and I. Procaccia,
       "Measuring the strangeness of strange attractors,"  *Physica 9d*, pp. 189-208.
    """
    if inp.ndim != 2:
        raise ValueError('Input array must be two-dimensional.')

    if mode == 'bader':
        cdim_func = _features.cdim_bader
        inp_: NDArray
        if inp.dtype == 'int16':
            inp_ = inp.copy()
        else:
            inp_ = fti16(inp)

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

    out = _np.zeros((inp.shape[1], n_delay, inp.shape[0]-wlen-n_delay), dtype=_np.float64)
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
    buff = _np.empty_like(sig, dtype=_np.float64)
    _np.abs(sig, out=buff)
    _np.square(buff, out=buff)
    total = _np.empty((1, buff.shape[1]))
    return _np.sum(buff, axis=0, dtype=_np.float64, out=total, keepdims=True)


def rms(sig: FloatArray) -> FloatArray:
    """Root mean square of time domain signal.

    Args:
        sig:  Time domain signal

    Returns:
        RMS of signal along first axis.
    """
    buff = _np.empty_like(sig, dtype=_np.float64)
    out = _np.empty((sig.shape[1], 1), dtype=_np.float64)
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
    out = _np.empty_like((1, amps.shape[1]), dtype=_np.float64)
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

    Args:
        inp:    Input data. Each row is assumend DFT bins.
        delta:  Sample spacing.
        total:  Accumulate over first axis.

    Returns:
        Array of Spectral flux.

    Note:
        Spextral flux is computed by

        .. math::
            SF(i) = \sum_{j=0}^k H(|X_{i,j}| - |X_{i-1,j}|) \,,

        where :math:`X_{i,j}` is the :math:`j` th frequency bin of the :math:`i`
        th spectrum :math:`X` of a spectrogram :math:`\boldsymbol X`.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    out = _np.empty_like(inp, dtype=_np.float64)
    _np.maximum(_np.gradient(inp, delta, axis=-1), 0, out=out)
    if total:
        return floatarray(out.sum(axis=0, keepdims=True))
    return out


def spl(inp: FloatArray, ref: float = _defaults.SPL_REF) -> FloatArray:
    """Computes the average sound pressure level of time domain signal.

    Args:
        inp:  Time domain signal.
        ref:  Reference level.

    Returns:
        Average sound pressure level.
    """
    level = rms(inp)/ref
    _np.log10(level, where=level>0, out=level)
    _np.multiply(level, 20.0, out=level)
    return level


def log_attack_time(inp: FloatArray, fps: int, ons_idx: IntArray,
                    wlen: float = 0.05) -> FloatArray:
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
    return floatarray(_np.log(attack_time))


def loudness(frqs: FloatArray, bins: FloatArray) -> FloatArray:
    """Calculate a measure for the perceived loudness from a spectrogram.

    Args:
        frqs:   Frquency axis.
        bins:   Magnitude spectrogram.

    Returns:
        Estimate of the total loudness.
    """
    cbrs = _cb.filter_bank(frqs) @ bins
    return _cb.total_loudness(cbrs)


def roughness_helmholtz(d_frq: float, bins: FloatArray, frq_max: float,
                        total: bool = True) -> FloatArray:
    """Estimate auditory roughness using Helmholtz' algorithm.

    Args:
        d_frq:      Frequency spacing
        bin:        DFT bins
        frq_max:    Maximum frequency
        total:      If ``True``, return total roughness over time.

    Returns:
        Auditoory roughness
    """
    kernel = _roughnes_kernel(d_frq, frq_max)
    out = _np.empty((kernel.size, bins.shape[1]))
    for i, bin_slice in enumerate(bins.T):
        out[:, i] = _np.correlate(bin_slice, kernel, mode='same')

    if total is True:
        out = out.sum(axis=0, keepdims=True)
    return out


def sharpness(frqs: FloatArray, bins: FloatArray) -> FloatArray:
    """Calculate a measure for the perception of auditory sharpness from a
    spectrogram.

    Args:
        frqs:    Frequencies.
        bins:    DFT magnitudes.

    Returns:
        Sharpness.
    """
    cbrs = _cb.filter_bank(frqs.squeeze()) @ bins
    return _cb.sharpness(cbrs)


def _power_distr(bins: FloatArray) -> FloatArray:
    """Computes the spectral energy distribution.

    Args:
        bins:    NxM array of DFT bins.

    Returns:
        NxM array of spectral densities.
    """
    total_power = _np.empty((1, bins.shape[1]), dtype=_np.float64)
    _np.sum(bins, axis=0, keepdims=True, out=total_power)
    total_power[total_power == 0] = 1
    return bins / total_power


def _roughnes_kernel(frq_res: float, frq_max: float) -> FloatArray:
    """Comput the convolution kernel for roughness computation.

    Args:
        frq_res:    Frequency resolution
        frq_max:    Frequency bound.

    Returns:
        Weight for each frequency below ``frq_max``.
    """
    frm = 33.5
    bin_idx = int(round(frq_max/frq_res))
    norm = frm * _np.exp(-1)
    base = _np.abs(_np.arange(-bin_idx, bin_idx+1)) * frq_res

    out = _np.empty_like(base, dtype=_np.float64)
    _np.divide(base, norm, out=out)
    _np.multiply(out, _np.exp(-base/frm), out)
    return out
