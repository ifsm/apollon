import unittest

import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from apollon.signal.critical_bands import (filter_bank, frq2cbr, level,
                                           masking_slope, sharpness,
                                           specific_loudness, spread,
                                           total_loudness, weight_factor)


class TestFilterBank(unittest.TestCase):

    def test_filter_bank_dimensions(self):
        """Prüft, ob die Dimensionen der Filterbank korrekt sind."""
        frqs = np.linspace(0, 8000, 1000)
        fbank = filter_bank(frqs)
        
        z_max = np.ceil(frq2cbr(8000).max()).astype(int)
        self.assertEqual(fbank.shape[0], z_max)
        self.assertEqual(fbank.shape[1], 1000)

    @given(arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=1000),
        elements=st.floats(min_value=0, max_value=45000, allow_nan=False, allow_infinity=False)
    ))
    def test_filter_bank_properties(self, frqs):
        """Prüft allgemeine mathematische Eigenschaften der Filterbank."""
        frqs = np.sort(frqs)
        fbank = filter_bank(frqs)

        self.assertTrue(np.all(fbank >= 0))
        # Each band's window is rescaled so its *sum* equals its bin count,
        # not so its *peak* stays at 1 -- the peak approaches (but stays
        # below) 2 as a band's bin count grows.
        self.assertTrue(np.all(fbank <= 2))

        non_zero_counts = np.sum(fbank > 0, axis=0)
        self.assertTrue(np.all(non_zero_counts <= 1))

    def test_empty_input(self):
        """Prüft das Verhalten bei leerem Input."""
        frqs = np.array([])
        fbank = filter_bank(frqs)
        self.assertEqual(fbank.size, 0)

    @given(st.floats(min_value=0, max_value=100))
    def test_single_frequency(self, f):
        """Prüft, ob ein einzelner Frequenzwert ein korrektes Shape liefert."""
        frqs = np.array([f])
        fbank = filter_bank(frqs)
        self.assertEqual(fbank.shape[1], 1)
        if f > 0:
            self.assertGreaterEqual(fbank.shape[0], 1)

    def test_negative_frequency_raises(self):
        """A negative frequency must be rejected rather than silently
        dropped from every band."""
        with self.assertRaises(ValueError):
            filter_bank(np.array([-100.0, 1000.0]))


class TestFrq2cbr(unittest.TestCase):

    def test_negative_frequency_raises(self):
        """frq2cbr has no defined output for negative frequencies -- it
        must reject them rather than silently returning a nonsensical
        negative Bark rate."""
        with self.assertRaises(ValueError):
            frq2cbr(np.array([-1.0]))


class TestSharpness(unittest.TestCase):

    n_bands = 22

    def _single_band(self, band):
        """Return a one-frame spectrogram with energy in ``band`` only."""
        spctrm = np.zeros((self.n_bands, 1))
        spctrm[band] = 1.0
        return spctrm

    def test_single_band_regression(self):
        """A lone active band no longer has a closed form once excitation
        spreading (spread()) leaks its energy into neighbouring bands --
        pin the current, numerically verified output instead."""
        expected = {2: 1.454389706443187,
                    17: 4.023327825025507,
                    20: 5.312699429940025}
        for band, exp in expected.items():
            with self.subTest(band=band):
                self.assertAlmostEqual(sharpness(self._single_band(band))[0], exp)

    def test_frame_independence(self):
        """Each time instant is normalized by its own total loudness."""
        spctrm = np.random.default_rng(0).random((self.n_bands, 5)) * 1e-3
        per_frame = [sharpness(spctrm[:, i:i+1])[0] for i in range(5)]
        self.assertTrue(np.allclose(sharpness(spctrm), per_frame))

    def test_frame_count_invariance(self):
        """Appending time instants leaves the existing values untouched."""
        spctrm = np.random.default_rng(1).random((self.n_bands, 3)) * 1e-3
        duplicated = np.concatenate([spctrm, spctrm], axis=1)
        self.assertTrue(np.allclose(sharpness(duplicated)[:3], sharpness(spctrm)))

    def test_level_dependence(self):
        """Sharpness increases with level: masking spreads further upward
        in Bark at higher masker levels (masking_slope's level-dependent
        upper flank), so a louder spectrum of the same shape reads sharper."""
        spctrm = self._single_band(17)
        values = [sharpness(spctrm*scale)[0] for scale in (1e-6, 1e-4, 1e-2, 1e0, 1e2)]
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_output_shape(self):
        """One value per time instant, and a scalar for a single spectrum."""
        spctrm = np.random.default_rng(2).random((self.n_bands, 5)) * 1e-3
        self.assertEqual(sharpness(spctrm).shape, (5,))
        self.assertEqual(sharpness(spctrm[:, 0]).shape, ())

    def test_monotonic_in_band_position(self):
        """Moving the active band upwards increases sharpness."""
        values = [sharpness(self._single_band(band))[0]
                  for band in (2, 8, 14, 17, 20)]
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_weighting_onset(self):
        """The exponential weighting switches on above roughly 16 Bark."""
        self.assertEqual(weight_factor(np.array([15.5]))[0], 1.0)
        self.assertGreater(weight_factor(np.array([16.5]))[0], 1.0)


class TestMaskingSlope(unittest.TestCase):

    def test_zero_at_own_band(self):
        """A masker's excitation at its own band (dz=0) is unattenuated,
        regardless of its frequency or level."""
        for frq, lvl in [(200.0, 20.0), (1000.0, 60.0), (8000.0, 100.0)]:
            with self.subTest(frq=frq, level=lvl):
                sf = masking_slope(np.array([0.0]), np.array([frq]), np.array([lvl]))
                self.assertEqual(sf[0], 0.0)

    def test_lower_flank_constant(self):
        """Below the masker, the slope is a constant 27 dB/Bark, independent
        of the masker's frequency or level."""
        dz = np.array([-2.0])
        sf_a = masking_slope(dz, np.array([200.0]), np.array([20.0]))
        sf_b = masking_slope(dz, np.array([8000.0]), np.array([100.0]))
        self.assertAlmostEqual(sf_a[0], -54.0)
        self.assertAlmostEqual(sf_b[0], -54.0)

    def test_upper_flank_shallower_at_higher_level(self):
        """A louder masker's excitation reaches further upward (a shallower,
        less negative, upper-flank slope)."""
        dz = np.array([3.0])
        quiet = masking_slope(dz, np.array([1000.0]), np.array([20.0]))
        loud = masking_slope(dz, np.array([1000.0]), np.array([100.0]))
        self.assertGreater(loud[0], quiet[0])

    def test_upper_flank_shallower_than_lower(self):
        """At equal Bark distance, the upper flank is shallower than the
        lower flank -- masking spreads further upward than downward."""
        sf_up = masking_slope(np.array([2.0]), np.array([1000.0]), np.array([60.0]))
        sf_down = masking_slope(np.array([-2.0]), np.array([1000.0]), np.array([60.0]))
        self.assertGreater(sf_up[0], sf_down[0])


class TestSpreading(unittest.TestCase):

    n_bands = 22

    def test_silence_stays_silent(self):
        """An all-zero critical band spectrum spreads to all zero."""
        spread_out = spread(np.zeros(self.n_bands))
        self.assertTrue(np.all(spread_out == 0.0))

    def test_leaks_into_neighbours(self):
        """A single active band's energy leaks into neighbouring bands
        after spreading."""
        frame = np.zeros(self.n_bands)
        frame[10] = 1e-6
        spread_out = spread(frame)
        self.assertGreater(spread_out[9], 0.0)
        self.assertGreater(spread_out[11], 0.0)

    def test_self_band_preserved(self):
        """A band's own contribution to itself is unattenuated."""
        frame = np.zeros(self.n_bands)
        frame[10] = 1e-6
        spread_out = spread(frame)
        self.assertAlmostEqual(spread_out[10], 1e-6)

    def test_spreads_further_upward_than_downward(self):
        """The excitation pattern falls off more slowly above an active
        band than below it, matching masking_slope's asymmetry."""
        frame = np.zeros(self.n_bands)
        frame[10] = 1e-6
        spread_out = spread(frame)
        self.assertGreater(spread_out[13], spread_out[7])

    def test_operates_per_frame(self):
        """Each time instant is spread independently of the others."""
        spctrm = np.random.default_rng(3).random((self.n_bands, 4)) * 1e-6
        per_frame = np.stack([spread(spctrm[:, i]) for i in range(4)], axis=1)
        self.assertTrue(np.allclose(spread(spctrm), per_frame))


class TestSpecificLoudness(unittest.TestCase):

    # Mirrors the ``ref`` constant hard-coded inside ``level()``.
    ref = 1e-12

    def test_level_zero_at_reference(self):
        """The reference intensity maps to a 0 dB critical band level."""
        self.assertAlmostEqual(level(np.array([self.ref]))[0], 0.0)

    def test_ten_db_step_constant_ratio(self):
        """A constant dB step yields a constant loudness ratio at any level."""
        bases = self.ref * np.array([1e0, 1e2, 1e4, 1e6])
        ratios = specific_loudness(bases*10) / specific_loudness(bases)
        self.assertTrue(np.allclose(ratios, 10**0.23))

    def test_specific_loudness_monotonic(self):
        """Specific loudness strictly increases with intensity."""
        intensities = self.ref * np.logspace(-6, 2, num=20)
        values = specific_loudness(intensities)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_total_loudness_responsive_across_range(self):
        """Total loudness spans orders of magnitude over a wide dB range,
        unlike the old dB-level-exponent form which barely moved."""
        low = total_loudness(np.array([[self.ref]]))[0]
        high = total_loudness(np.array([[self.ref * 10**12]]))[0]
        self.assertAlmostEqual(high/low, 10**(12*0.23), delta=1.0)
