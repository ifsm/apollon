import numpy as _np
from scipy.signal import hilbert as _hilbert

from .. import segment as _segment
from .. tools import array2d_fsum
from .. types import Array as _Array
from .  import critical_bands as _cb
from .  roughness import roughness
from .  tools import trim_spectrogram

def spectral_centroid(inp: _Array, frqs: _Array) -> _Array:
    """Estimate the spectral centroid frequency.

    Calculation is applied to the second axis. One-dimensional
    arrays will be promoted.

    Args:
        inp  (ndarray)    Input. Each row is assumend FFT bins scaled by `freqs`.
        frqs (ndarray)    One-dimensional array of FFT frequencies.

    Returns:
        (ndarray or scalar)    Spectral centroid frequency.
    """
    inp = _np.atleast_2d(inp).astype('float64')

    weighted_nrgy = _np.multiply(inp, frqs).sum(axis=1).squeeze()
    total_nrgy = inp.sum(axis=1).squeeze()
    total_nrgy[total_nrgy == 0.0] = 1.0

    return _np.divide(weighted_nrgy, total_nrgy)


def spectral_flux(inp: _Array, delta:float=1.0) -> _Array:
    """Estimate the spectral flux

    Args:
        inp   (ndarray)    Input. Each row is assumend FFT bins.
        delta (float)      Sample spacing.

    Returns:
        (ndarray)    Spectral flux.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    return _np.maximum(_np.gradient(inp, delta, axis=-1), 0).squeeze()


def spectral_shape(inp, frqs, low: float = 50, high: float = 16000):
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
        inp  (ndarray)    Input spectrum or spectrogram.
        frqs (ndarray)    Frequency axis.
        low  (float)      Lower cutoff frequency.
        high (float)      Upper cutoff frequency.

    Returns:
        (tuple)    (centroid, spread, skewness, kurtosis)
    """

    if inp.ndim < 2:
        inp = inp[:, None]

    vals, frqs = trim_spectrogram(inp, frqs, 50, 16000)

    total_nrgy = array2d_fsum(vals)
    total_nrgy[total_nrgy == 0.0] = 1.0    # Total energy is zero iff input signal is all zero.
                                           # Replace these bin values with 1, so division by
                                           # total energy will not lead to nans.

    centroid = frqs @ vals / total_nrgy
    deviation = frqs[:, None] - centroid

    spread = array2d_fsum(_np.power(deviation, 2) * vals)
    skew   = array2d_fsum(_np.power(deviation, 3) * vals)
    kurt   = array2d_fsum(_np.power(deviation, 4) * vals)

    spread = _np.sqrt(spread / total_nrgy)
    skew   = skew / total_nrgy / _np.power(spread, 3)
    kurt   = kurt / total_nrgy / _np.power(spread, 4)

    return centroid, spread, skew, kurt


def log_attack_time(inp: _Array, fs: int, ons_idx: _Array, wlen:float=0.05) -> _Array:
    """Estimate the attack time of each onset an return is logarithm.

    This function estimates the attack time as the duration between the
    onset and the local maxima of the magnitude of the Hilbert transform
    of the local window.

    Args:
        inp     (ndarray)    Input signal.
        fs      (int)        Sampling frequency.
        ons_idx (ndarray)    Sample indices of onsets.
        wlen    (float)      Local window length in samples.

    Returns:
        (ndarray)    Logarithm of the attack time.
    """
    wlen = int(fs * wlen)
    segs = _segment.by_onsets(inp, wlen, ons_idx)
    mx = _np.absolute(_hilbert(segs)).argmax(axis=1) / fs
    mx[mx == 0.0] = 1.0

    return _np.log(mx)


def sharpness(inp: _Array, frqs: _Array) -> _Array:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram.

    Args:
        inp  (ndarray)    Two-dimensional input array. Assumed to be an magnitude spectrogram.
        frqs (ndarray)    Frequency axis of the spectrogram.

    Returns:
        (ndarray)    Sharpness for each time instant of the spectragram.
    """
    cbrs = _cb.filter_bank(frqs) @ inp
    return _cb.sharpness(cbrs)
