from unittest import TestCase
import unittest

from hypothesis import strategies as st
from hypothesis import assume, given

from apollon.signal.cepstral import Mfcc
from apollon.signal.spectral import Dft
from apollon.signal.filter import TriangFilterSpec

from .strategies import frequencies, samplerates, fftsizes

filts = st.integers(min_value=1, max_value=100)


class TestMfcc(TestCase):
    def setUp(self) -> None:
        self.fps = 44100
        self.n_fft = 44100
        self.lcf = 80
        self.ucf = 5000
        self.n_filters = 50
        self.window = "hamming"

    def test__mfcc(self) -> None:
        Mfcc(self.fps, self.n_fft, self.lcf, self.ucf, self.n_filters)

    def test__mfcc_from_stft(self) -> None:
        dft = Dft(self.fps, self.window, self.n_fft)
        spec = TriangFilterSpec(low=self.lcf, high=self.ucf, n_filters=self.n_filters)
        mfcc = Mfcc.from_dft(dft, spec)
        self.assertIsInstance(mfcc, Mfcc)
        self.assertEqual(mfcc.fps, dft.params.fps)
        self.assertEqual(mfcc.n_fft, dft.params.n_fft)
