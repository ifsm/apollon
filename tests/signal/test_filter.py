from unittest import TestCase

from apollon.typing import FloatArray
from apollon.signal.filter import triang, triang_filter_bank
from hypothesis import strategies as st
from hypothesis import assume, given
import numpy as np

from .strategies import filterspecs, triangspec


class TestTriang(TestCase):
    @given(filterspecs(), st.just((0, 1, 0)))
    def test_triang(
        self, filterspec: tuple[int, int, FloatArray], amps: tuple[float, float, float]
    ) -> None:
        fps, size, frqs = filterspec
        res = triang(fps, size, frqs, amps)
        self.assertEqual(res.shape[0], frqs.shape[0])

    @given(st.integers(min_value=0, max_value=2))
    def test_bad_nfft(self, n_fft: int) -> None:
        with self.assertRaises(ValueError):
            triang(1, n_fft, np.array([[0, 1, 2]]))


class TestTriangFilterBank(TestCase):
    @given(triangspec())
    def test_triang_filter_bank(self, spec: tuple[float, float, int, int, int]) -> None:
        low, high, n_filters, fps, size = spec
        if low < high:
            if high > fps//2:
                with self.assertRaises(ValueError):
                    fb = triang_filter_bank(low, high, n_filters, fps, size)
            else:
                fb = triang_filter_bank(low, high, n_filters, fps, size)
                self.assertIsInstance(fb, np.ndarray)
        else:
            with self.assertRaises(ValueError):
                fb = triang_filter_bank(low, high, n_filters, fps, size)
                self.assertIsInstance(fb, np.ndarray)
