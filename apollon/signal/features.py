import numpy as _np
from scipy.signal import hilbert as _hilbert

from .. import segment as _segment
from .. types import Array as _Array


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

def spectral_flux(inp: _Array) -> _Array:
    """Estimate the spectral flux

    Args:
        inp (ndarray)    Input. Each row is assumend FFT bins.

    Returns:
        (ndarray)    Spectral flux.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    return _np.maximum(_np.diff(inp, axis=-1), 0).squeeze()


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
