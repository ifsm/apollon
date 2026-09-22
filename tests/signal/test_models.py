import unittest

from pydantic import ValidationError

from apollon.signal.models import (CepstrumParams, DftParams, StftParams,
                                   TriangFilterSpec)
from apollon.signal.spectral import Stft


class TestStftParams(unittest.TestCase):
    valid = {'fps': 44100, 'n_perseg': 1024, 'n_overlap': 512,
             'extend': True, 'pad': True}

    def test_accepts_valid_params(self):
        StftParams(**self.valid)
        StftParams(**self.valid, n_fft=2048)
        StftParams(**{**self.valid, 'n_overlap': 0})

    def test_rejects_invalid_params(self):
        for change in ({'fps': 0}, {'fps': -44100}, {'n_fft': 0},
                       {'n_fft': 512}, {'n_perseg': 0}, {'n_overlap': -1},
                       {'n_overlap': 1024}):
            with self.subTest(**change):
                with self.assertRaises(ValidationError):
                    StftParams(**{**self.valid, **change})

    def test_stft_fails_at_construction(self):
        """An FFT shorter than a segment fails before any transform."""
        with self.assertRaises(ValueError):
            Stft(fps=44100, n_perseg=64, n_overlap=32, n_fft=16)


class TestDftParams(unittest.TestCase):
    def test_rejects_non_positive_values(self):
        for kwargs in ({'fps': 0}, {'fps': 44100, 'n_fft': 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValidationError):
                    DftParams(**kwargs)


class TestCepstralParams(unittest.TestCase):
    def test_rejects_negative_lifter_gain(self):
        with self.assertRaises(ValidationError):
            CepstrumParams(lifter_gain=-1.0)

    def test_zero_lifter_gain_is_valid(self):
        self.assertEqual(CepstrumParams(lifter_gain=0.0).lifter_gain, 0.0)

    def test_filter_spec_rejects_invalid_values(self):
        for kwargs in ({'low': -10.0, 'high': 1000.0, 'n_filters': 4},
                       {'low': 0.0, 'high': 1000.0, 'n_filters': 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValidationError):
                    TriangFilterSpec(**kwargs)


if __name__ == '__main__':
    unittest.main()
