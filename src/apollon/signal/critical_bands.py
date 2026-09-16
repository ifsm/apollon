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


def _band_frequencies(bark: FloatArray) -> FloatArray:
    """Approximate the centre frequency (Hz) for each critical band rate.

    Numerically inverts ``frq2cbr`` against a dense forward table, rather
    than using a different author's closed-form approximation, so the
    inverse stays exactly consistent with this module's own Bark formula.

    Args:
        bark: Critical band rate(s) in Bark.

    Returns:
        Approximate centre frequency in Hz.
    """
    hz_grid = _np.linspace(0.0, 30000.0, 30001)
    bark_grid = frq2cbr(hz_grid)
    return floatarray(_np.interp(bark, bark_grid, hz_grid))


def masking_slope(dz: FloatArray, frq: FloatArray,
                   masker_level: FloatArray) -> FloatArray:
    """Return the Terhardt masking-slope excitation level (dB) at Bark
    distance ``dz`` from a masker.

    The excitation a masker produces in a neighbouring critical band falls
    off from the masker's own level at a rate that is constant below the
    masker (27 dB/Bark) but shallower, and level- and frequency-dependent,
    above it -- masking spreads further upward in frequency than downward.
    Same source as ``specific_loudness``'s Zwicker & Fastl (1999) citation.

    Args:
        dz: Bark distance from the masker (``z_target - z_masker``).
        frq: Masker centre frequency in Hz.
        masker_level: Masker's own critical band level in dB.

    Returns:
        Excitation level (dB) relative to the masker's own level.
    """
    # A silent masker (masker_level == -inf) always contributes zero energy
    # downstream regardless of its slope, so substitute a finite placeholder
    # here to avoid an -inf * 0 = nan at dz == 0 rather than let it propagate.
    safe_level = _np.where(_np.isneginf(masker_level), 0.0, masker_level)
    lower = 27.0 * dz
    upper = (-24.0 - 230.0/frq + 0.2*safe_level) * dz
    return floatarray(_np.where(dz <= 0, lower, upper))


def spread(cbr: FloatArray, resolution: float = 1.0) -> FloatArray:
    """Apply excitation spreading across critical bands.

    Models auditory-filter leakage between neighbouring Bark bands: each
    band's energy leaks into its neighbours according to
    ``masking_slope``, turning the per-band critical-band intensity
    pattern into an excitation pattern. Because the upward slope depends on
    each masker band's own level, this is applied one time instant at a
    time.

    Args:
        cbr: Critical band rate spectrum (intensity/power).
        resolution: Bark width of one row of ``cbr``. Defaults to the
            standard 1-Bark critical band; pass a smaller value when
            ``cbr`` is a finer-grained excitation pattern (see
            ``excitation_pattern``).

    Returns:
        Excitation pattern: ``cbr`` after cross-band masking spread.

    Raises:
        ValueError: If ``resolution`` is not positive.
    """
    if resolution <= 0:
        raise ValueError('resolution must be positive.')
    cbr = _np.asarray(cbr, dtype='float64')
    n_bands = cbr.shape[0]
    z = (_np.arange(n_bands, dtype='float64') + 0.5) * resolution
    frqs = _band_frequencies(z)
    dz = z[:, None] - z[None, :]   # dz[j, i] = distance from masker i to band j

    def _spread_frame(frame: FloatArray) -> FloatArray:
        masker_level = level(frame)
        sf_db = masking_slope(dz, frqs[None, :], masker_level[None, :])
        gain = _np.power(10.0, sf_db/10.0)
        return gain @ frame

    if cbr.ndim == 1:
        return floatarray(_spread_frame(cbr))
    return floatarray(_np.stack([_spread_frame(cbr[:, t])
                                  for t in range(cbr.shape[1])], axis=1))


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


def total_loudness(cbr: FloatArray, spread_input: bool = True) -> FloatArray:
    """Compute the totals loudness of critical band rate spectra.

    The total loudness is the sum of the specific loudnesses, computed on
    the excitation pattern after cross-band masking spread (``spread``),
    not on the raw per-band intensities directly.

    Args:
        cbr: Critical band rate spectrum (intensity/power).
        spread_input: If ``True`` (default), apply ``spread`` (at the
            standard 1-Bark resolution) to ``cbr`` first. Set ``False``
            when ``cbr`` is already a prepared excitation pattern (e.g.
            from ``excitation_pattern``), to avoid spreading twice.

    Returns:
        Total loudness
    """
    if spread_input:
        cbr = spread(cbr)
    return floatarray(specific_loudness(cbr).sum(axis=0))


def filter_bank(frqs: FloatArray, resolution: float = 1.0) -> FloatArray:
    """Return a critical band rate scaled filter bank.

    Each filter is triangular, which lower and upper cuttoff frequencies
    set to lower and upper bound of the given critical band rate.

    Row ``i`` of the returned filter bank always corresponds to Bark band
    ``i`` (of width ``resolution``). A band with no frequency bin in
    ``frqs`` gets an all-zero row rather than being omitted, so the row
    count and row-to-band mapping don't depend on how densely ``frqs``
    happens to sample the Bark scale.

    Each band's triangular window is rescaled to sum to exactly the number
    of bins it contains, so a band's total gain on a flat spectrum scales
    with its bin count rather than fluctuating with the bin count's parity.

    Args:
        frqs:   Frequency axis in Hz
        resolution: Bark width of one row. Defaults to the standard 1-Bark
            critical band; pass a smaller value to build a finer-grained
            filter bank (see ``excitation_pattern``).

    Returns:
        Bark scaled filter bank

    Raises:
        ValueError: If ``frqs`` contains negative frequencies (propagated
            from ``frq2cbr``), or if ``resolution`` is not positive.
    """
    if resolution <= 0:
        raise ValueError('resolution must be positive.')
    z_frq = frq2cbr(frqs) / resolution
    bands = z_frq.astype(int)
    n_bands = int(bands.max()) + 1 if bands.size else 0
    fbank = _np.zeros((n_bands, z_frq.size))

    for bnd in range(n_bands):
        idx, = _np.nonzero(bands==bnd)
        if idx.size:
            window = _get_window('triang', idx.size, False)
            fbank[bnd, idx] = window * (idx.size / window.sum())

    return fbank


def _coarsen(fine: FloatArray, resolution: float) -> FloatArray:
    """Sum a fine-resolution critical band pattern back down into standard
    1-Bark-wide bands.

    Args:
        fine: Pattern at ``resolution``-Bark row width (one frame,
            ``(n_fine,)``, or a spectrogram, ``(n_fine, T)``).
        resolution: Bark width of one row of ``fine``.

    Returns:
        Pattern at standard 1-Bark row width.
    """
    n_fine = fine.shape[0]
    if n_fine == 0:
        return floatarray(fine)
    fine_centres = (_np.arange(n_fine, dtype='float64') + 0.5) * resolution
    n_bands = int(fine_centres[-1]) + 1
    coarse_idx = _np.clip(fine_centres.astype(int), 0, n_bands - 1)
    coarse = _np.zeros((n_bands,) + fine.shape[1:])
    _np.add.at(coarse, coarse_idx, fine)
    return floatarray(coarse)


def excitation_pattern(frqs: FloatArray, power: FloatArray,
                        resolution: float = 0.1) -> FloatArray:
    """Compute the 1-Bark critical band excitation pattern from a raw power
    spectrum.

    Spreads at finer-than-1-Bark resolution before aggregating down to
    standard 1-Bark critical bands -- matching the order of operations the
    full Zwicker model uses (spread first, then bin), rather than spreading
    energy that has already been coarsely quantized into 1-Bark bands.

    Args:
        frqs: Frequency axis in Hz.
        power: Power spectrum (or spectrogram) aligned with ``frqs``.
        resolution: Bark width of the fine spreading grid. Smaller values
            track DIN 45692's calibration more closely, at higher
            computational cost (an O(n^2) masking-spread matrix per time
            instant, with n inversely proportional to ``resolution``).

    Returns:
        Critical band rate spectrum at standard 1-Bark resolution, ready
        for ``total_loudness``/``sharpness`` (called with
        ``spread_input=False``, since spreading already happened here).
    """
    fine = filter_bank(frqs, resolution) @ power
    return _coarsen(spread(fine, resolution), resolution)


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


def sharpness(cbr_spctrm: FloatArray, spread_input: bool = True) -> FloatArray:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram
    of critical band levels.

    Row ``i`` of ``cbr_spctrm`` is taken to be Bark band ``i``, whose
    representative critical band rate is the band centre ``i + 0.5``. Cross-band
    masking spread (``spread``) is applied first, so specific loudness weights
    both the numerator and the denominator of an excitation pattern rather than
    the raw per-band intensities. The result is a weighted mean of the critical
    band rate and hence independent of the overall level and of the number of
    time instants. The ``0.11`` scaling constant of the Peeters/DIN 45692
    sharpness formulation is applied (same source as ``weight_factor``'s
    Peeters citation, section 8.1.3).

    Args:
        cbr_spctrm: Critical band rate Spectrogram
        spread_input: If ``True`` (default), apply ``spread`` (at the
            standard 1-Bark resolution) to ``cbr_spctrm`` first. Set
            ``False`` when ``cbr_spctrm`` is already a prepared excitation
            pattern (e.g. from ``excitation_pattern``), to avoid spreading
            twice.

    Returns:
        Sharpness for each time instant of the ``cbr_spctrm``.
    """
    if spread_input:
        cbr_spctrm = spread(cbr_spctrm)
    loud_specific = _np.maximum(specific_loudness(cbr_spctrm), _np.finfo('float64').eps) # pylint: disable=E1101
    loud_total = loud_specific.sum(axis=0)

    cbrs = _np.arange(cbr_spctrm.shape[0], dtype='float64') + 0.5
    return floatarray(0.11 * ((cbrs * weight_factor(cbrs)) @ loud_specific) / loud_total)
