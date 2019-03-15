import numpy as _np
from scipy.signal.windows import get_window as _get_window

from .. types import Array as _Array


def frq2cbr(frq) -> _Array:
    """Transform frquencies in Hz to critical band rates in Bark.

    Args:
        frq    Frequency in Hz.

    Returns:
        Critical band rate.
    """
    frq = _np.atleast_1d(frq)
    return 13.0 * _np.arctan(0.00076*frq) + 3.5 * _np.arctan(_np.power(frq/7500, 2))


def level(cbi:_Array):
    """Compute the critical band level L_G from critical band intensities I_G.

    Args:
        cbi (ndarray)    Critical band intensities.

    Returns:
        (nddarray)    Critical band levels.
    """
    ref = 10e-12
    return 10.0 * _np.log10(_np.maximum(cbi, ref) / ref)


def specific_loudness(cbr:_Array):
    """Compute the specific loudness of a critical band rate spectra.

    The specific loudness is the loudness per critical band rate. The spectra
    should be scaled in critical band levels.

    Args:
        cbr (ndarray)    Critical band rate spectrum.

    Returns:
        (ndarray)    Specific loudness.
    """
    return _np.power(level(cbr), 0.23)


def total_loudness(cbr:_Array):
    """Compute the totals loudness of critical band rate spectra.

    The total loudness is the sum of the specific loudnesses. The spectra
    should be scaled to critical band levels.

    Args:
        cbr_spctr (ndarray)    Critical band rate spectra.

    Returns:
        (ndarray)    Total loudness.
    """
    return specific_loudness(cbr).sum(axis=0)


def filter_bank(frqs):
    """Return a critical band rate scaled filter bank.

    Each filter is triangular, which lower and upper cuttoff frequencies
    set to lower and upper bound of the given critical band rate.

    Args:
        frqs (ndarray)    Frequency axis in Hz.

    Returns:
        (ndarray)    Bark scaled filter bank.
    """
    n_bands = 24
    z_frq = frq2cbr(frqs)
    filter_bank = _np.zeros((n_bands, z_frq.size))

    for z in range(n_bands):
        lo = z
        hi = z + 1

        idx = _np.logical_and(lo <= z_frq, z_frq < hi)
        n = idx.sum()
        filter_bank[lo, idx] = _get_window('triang', n, False)

    return filter_bank


def weight_factor(z):
    """Return weighting factor per critical band rate for sharpness calculation.

    This is an improved version of Peeters (2004), section 8.1.3.

    Args:
        z (ndarray)    Critical band rate.

    Returns:
        (ndarray)    Wheighting factor.
    """
    base = _np.ones_like(z, dtype='float64')
    slope = 0.066 * _np.exp(0.171 * _np.atleast_1d(z))

    return _np.maximum(base, slope)
