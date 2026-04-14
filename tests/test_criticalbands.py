import unittest

import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from apollon.signal.critical_bands import filter_bank, frq2cbr


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
        self.assertTrue(np.all(fbank <= 1))

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
