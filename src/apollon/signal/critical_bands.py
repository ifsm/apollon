"""
Critical band helpers
"""
from typing import cast

import numpy as _np
from scipy.signal.windows import get_window as _get_window

from .. typing import FloatArray, floatarray
from .. import tools as _tools


def frq2cbr(frq: FloatArray) -> FloatArray:
    """Transform frquencies in Hz to critical band rates in Bark.

    Args:
        frq: Frequency in Hz

    Returns:
        Critical band rate
    """
    frq = _np.atleast_1d(frq)
    part1 = 13.0 * _np.arctan(0.00076*frq)
    part2 = 3.5 * _np.arctan(_np.power(frq/7500, 2))
    return floatarray(part1+part2)


def level(cbi: FloatArray) -> FloatArray:
    """Compute the critical band level L_G from critical band intensities I_G.

    Args:
        cbi: Critical band intensities

    Returns:
        Critical band levels
    """
    ref = 10e-12
    return floatarray(10.0 * _np.log10(_np.maximum(cbi, ref) / ref))


def specific_loudness(cbr: FloatArray) -> FloatArray:
    """Compute the specific loudness of a critical band rate spectrum.

    The specific loudness is the loudness per critical band rate. The spectra
    should be scaled in critical band levels.

    Args:
        cbr: Critical band rate spectrum

    Returns:
        Specific loudness
    """
    return _np.power(level(cbr), 0.23)


def total_loudness(cbr: FloatArray) -> FloatArray:
    """Compute the totals loudness of critical band rate spectra.

    The total loudness is the sum of the specific loudnesses. The spectra
    should be scaled to critical band levels.

    Args:
        cbr_spctr: Critical band rate spectra.

    Returns:
        Total loudness
    """
    return floatarray(specific_loudness(cbr).sum(axis=0))


def filter_bank(frqs: FloatArray) -> FloatArray:
    """Return a critical band rate scaled filter bank.

    Each filter is triangular, which lower and upper cuttoff frequencies
    set to lower and upper bound of the given critical band rate.

    Row ``i`` of the returned filter bank always corresponds to Bark band
    ``i``. A band with no frequency bin in ``frqs`` gets an all-zero row
    rather than being omitted, so the row count and row-to-band mapping
    don't depend on how densely ``frqs`` happens to sample the Bark scale.

    Each band's triangular window is rescaled to sum to exactly the number
    of bins it contains, so a band's total gain on a flat spectrum scales
    with its bin count rather than fluctuating with the bin count's parity.

    Args:
        frqs:   Frequency axis in Hz

    Returns:
        Bark scaled filter bank
    """
    z_frq = frq2cbr(frqs)
    bands = z_frq.astype(int)
    n_bands = int(bands.max()) + 1 if bands.size else 0
    fbank = _np.zeros((n_bands, z_frq.size))

    for bnd in range(n_bands):
        idx, = _np.nonzero(bands==bnd)
        if idx.size:
            window = _get_window('triang', idx.size, False)
            fbank[bnd, idx] = window * (idx.size / window.sum())

    return fbank


def weight_factor(cbr: FloatArray) -> FloatArray:
    """Return weighting factor per critical band rate for sharpness calculation.

    This is an improved version of Peeters (2004), section 8.1.3.

    Args:
        cbr: Critical band rate in Bark

    Returns:
        Weighting factor
    """
    base = _np.ones_like(cbr, dtype='float64')
    slope = 0.066 * _np.exp(0.171 * _np.atleast_1d(cbr))
    return cast(FloatArray, _np.maximum(base, slope))


def sharpness(cbr_spctrm: FloatArray) -> FloatArray:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram
    of critical band levels.

    Row ``i`` of ``cbr_spctrm`` is taken to be Bark band ``i``, whose
    representative critical band rate is the band centre ``i + 0.5``. Specific
    loudness weights both the numerator and the denominator, so the result is a
    weighted mean of the critical band rate and hence independent of the overall
    level and of the number of time instants. The ``0.11`` scaling constant of
    the Peeters/DIN 45692 sharpness formulation is applied.

    Args:
        cbr_spctrm: Critical band rate Spectrogram

    Returns:
        Sharpness for each time instant of the ``cbr_spctrm``.
    """
    loud_specific = _np.maximum(specific_loudness(cbr_spctrm), _np.finfo('float64').eps) # pylint: disable=E1101
    loud_total = loud_specific.sum(axis=0)

    cbrs = _np.arange(cbr_spctrm.shape[0], dtype='float64') + 0.5
    return floatarray(0.11 * ((cbrs * weight_factor(cbrs)) @ loud_specific) / loud_total)
