import unittest

import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from apollon.signal.critical_bands import (filter_bank, frq2cbr, sharpness,
                                           weight_factor)


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


class TestSharpness(unittest.TestCase):

    n_bands = 22

    def _single_band(self, band):
        """Return a one-frame spectrogram with energy in ``band`` only."""
        spctrm = np.zeros((self.n_bands, 1))
        spctrm[band] = 1.0
        return spctrm

    def test_single_band_closed_form(self):
        """A lone active band yields 0.11 * z * g(z) at that band's centre."""
        for band in (2, 17, 20):
            with self.subTest(band=band):
                centre = band + 0.5
                expected = 0.11 * centre * weight_factor(np.array([centre]))[0]
                self.assertAlmostEqual(sharpness(self._single_band(band))[0],
                                       expected)

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

    def test_level_invariance(self):
        """Sharpness is a weighted mean and hence invariant to overall level."""
        spctrm = self._single_band(17)
        values = [sharpness(spctrm*scale)[0] for scale in (1e-6, 1e-4, 1e-2, 1e0, 1e2)]
        self.assertLess(np.ptp(values), 1e-12)

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
