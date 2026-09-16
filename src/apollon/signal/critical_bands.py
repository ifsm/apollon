"""
Critical band helpers
"""
import numpy as _np
from scipy.signal.windows import get_window as _get_window

from .. typing import FloatArray, floatarray


def frq2cbr(frq: FloatArray) -> FloatArray:
    """Transform frquencies in Hz to critical band rates in Bark.

    Uses the Zwicker/Terhardt approximation [ZwickerTerhardt1980]_.

    Args:
        frq: Frequency in Hz

    Returns:
        Critical band rate

    Raises:
        ValueError: If ``frq`` contains negative frequencies.

    .. [ZwickerTerhardt1980] E. Zwicker, and E. Terhardt, "Analytical
       expressions for critical-band rate and critical bandwidth as a
       function of frequency," *Journal of the Acoustical Society of
       America*, 68(5), pp. 1523-1525.
    """
    frq = _np.atleast_1d(frq)
    if _np.any(frq < 0):
        raise ValueError('Frequencies must be non-negative.')
    part1 = 13.0 * _np.arctan(0.00076*frq)
    part2 = 3.5 * _np.arctan(_np.power(frq/7500, 2))
    return floatarray(part1+part2)


def level(cbi: FloatArray) -> FloatArray:
    """Compute the critical band level L_G from critical band intensities I_G.

    Args:
        cbi: Critical band intensities, i.e. proportional to physical power,
            not a pressure amplitude.

    Returns:
        Critical band levels. Zero (silent) input maps to ``-inf`` rather
        than a floor at the reference, so ``specific_loudness`` in turn maps
        silence to zero loudness instead of a constant, non-physical floor.
    """
    # Reference intensity, ISO 226 / Zwicker & Fastl (1999) I_0 = 1e-12 W/m^2.
    ref = 1e-12
    ratio = _np.maximum(cbi, 0.0) / ref
    out = _np.full_like(ratio, -_np.inf, dtype='float64')
    return floatarray(10.0 * _np.log10(ratio, where=ratio > 0, out=out))


def specific_loudness(cbr: FloatArray) -> FloatArray:
    """Compute the specific loudness of a critical band rate spectrum.

    The specific loudness is the loudness per critical band rate, following
    the power-law form of Zwicker's loudness model [ZwickerFastl1999]_,
    standardized in DIN 45692: it scales with the 0.23 power of the critical
    band *intensity ratio* (not the dB level itself), so a constant dB step
    produces a constant multiplicative change in loudness. ``cbr`` should be
    critical band intensities (i.e. power), consistent with ``level()``.

    Args:
        cbr: Critical band rate spectrum (intensity/power).

    Returns:
        Specific loudness

    .. [ZwickerFastl1999] E. Zwicker, and H. Fastl, *Psychoacoustics: Facts
       and Models*, 2nd ed., Springer, 1999.
    """
    # (cbi/ref)**0.23 == (10**(level/10))**0.23 == 10**(0.023*level)
    return floatarray(_np.power(10.0, 0.023 * level(cbr)))


def total_loudness(cbr: FloatArray) -> FloatArray:
    """Compute the totals loudness of critical band rate spectra.

    The total loudness is the sum of the specific loudnesses. The spectra
    should be scaled to critical band levels.

    Args:
        cbr: Critical band rate spectrum (intensity/power).

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

    Raises:
        ValueError: If ``frqs`` contains negative frequencies (propagated
            from ``frq2cbr``).
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

    This is an improved version of the sharpness weighting curve from
    Peeters [Peeters2004]_, section 8.1.3.

    Args:
        cbr: Critical band rate in Bark

    Returns:
        Weighting factor

    .. [Peeters2004] G. Peeters, "A large set of audio features for sound
       description (timbral and perceptual) in the CUIDADO project,"
       IRCAM technical report.
    """
    base = _np.ones_like(cbr, dtype='float64')
    slope = 0.066 * _np.exp(0.171 * _np.atleast_1d(cbr))
    return floatarray(_np.maximum(base, slope))


def sharpness(cbr_spctrm: FloatArray) -> FloatArray:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram
    of critical band levels.

    Row ``i`` of ``cbr_spctrm`` is taken to be Bark band ``i``, whose
    representative critical band rate is the band centre ``i + 0.5``. Specific
    loudness weights both the numerator and the denominator, so the result is a
    weighted mean of the critical band rate and hence independent of the overall
    level and of the number of time instants. The ``0.11`` scaling constant of
    the Peeters/DIN 45692 sharpness formulation is applied (same source as
    ``weight_factor``'s Peeters citation, section 8.1.3).

    Args:
        cbr_spctrm: Critical band rate Spectrogram

    Returns:
        Sharpness for each time instant of the ``cbr_spctrm``.
    """
    loud_specific = _np.maximum(specific_loudness(cbr_spctrm), _np.finfo('float64').eps) # pylint: disable=E1101
    loud_total = loud_specific.sum(axis=0)

    cbrs = _np.arange(cbr_spctrm.shape[0], dtype='float64') + 0.5
    return floatarray(0.11 * ((cbrs * weight_factor(cbrs)) @ loud_specific) / loud_total)
