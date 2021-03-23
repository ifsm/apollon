# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

import numpy as _np
from scipy.signal.windows import get_window as _get_window

from apollon.types import Array
from .. import tools as _tools


def frq2cbr(frq: Array) -> Array:
    """Transform frquencies in Hz to critical band rates in Bark.

    Args:
        frq:    Frequency in Hz.

    Returns:
        Critical band rate.
    """
    frq = _np.atleast_1d(frq)
    prod_a = 13.0 * _np.arctan(0.00076*frq)
    prod_b = 3.5 * _np.arctan(_np.power(frq/7500, 2))
    return prod_a + prod_b


def level(cbi: Array):
    """Compute the critical band level L_G from critical band intensities I_G.

    Args:
        cbi:    Critical band intensities.

    Returns:
        Critical band levels.
    """
    ref = 10e-12
    return 10.0 * _np.log10(_np.maximum(cbi, ref) / ref)


def specific_loudness(cbr: Array):
    """Compute the specific loudness of a critical band rate spectra.

    The specific loudness is the loudness per critical band rate. The spectra
    should be scaled in critical band levels.

    Args:
        cbr:    Critical band rate spectrum.

    Returns:
        Specific loudness.
    """
    return _np.power(level(cbr), 0.23)


def total_loudness(cbr: Array) -> Array:
    """Compute the totals loudness of critical band rate spectra.

    The total loudness is the sum of the specific loudnesses. The spectra
    should be scaled to critical band levels.

    Args:
        cbr_spctr:  Critical band rate spectra.

    Returns:
        Total loudness.
    """
    return _tools.fsum(specific_loudness(cbr), axis=0)


def filter_bank(frqs: Array) -> Array:
    """Return a critical band rate scaled filter bank.

    Each filter is triangular, which lower and upper cuttoff frequencies
    set to lower and upper bound of the given critical band rate.

    Args:
        frqs:    Frequency axis in Hz.

    Returns:
        Bark scaled filter bank.
    """
    n_bands = 24
    z_frq = frq2cbr(frqs)
    fbank = _np.zeros((n_bands, z_frq.size))

    for z in range(n_bands):
        lbf = z
        ubf = z + 1

        idx = _np.logical_and(lbf <= z_frq, z_frq < ubf)
        n = idx.sum()
        fbank[lbf, idx] = _get_window('triang', n, False)
    return fbank


def weight_factor(z):
    """Return weighting factor per critical band rate for sharpness calculation.

    This is an improved version of Peeters (2004), section 8.1.3.

    Args:
        z: Critical band rate.

    Returns:
        Weighting factor.
    """
    base = _np.ones_like(z, dtype='float64')
    slope = 0.066 * _np.exp(0.171 * _np.atleast_1d(z))
    return _np.maximum(base, slope)


def sharpness(cbr_spctrm: Array) -> Array:
    """Calculate a measure for the perception of auditory sharpness from a
    spectrogram of critical band levels.

    Args:
        cbr_spctrm: Critical band rate Spectrogram.

    Returns:
        Sharpness for each time instant of the ``cbr_spctrm``.
    """
    loud_specific = _np.maximum(specific_loudness(cbr_spctrm),
                                _np.finfo('float64').eps)
    loud_total = _tools.fsum(loud_specific, keepdims=True)

    z = _np.arange(1, 25)
    return ((z * weight_factor(z)) @ cbr_spctrm) / loud_total
