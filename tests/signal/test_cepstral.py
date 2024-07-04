from unittest import TestCase
import unittest

from hypothesis import strategies as st
from hypothesis import assume, given

from apollon.signal.cepstral import Mfcc


frqs = st.floats(min_value=0, max_value=10000, allow_infinity=False, allow_nan=False)
filts = st.integers(min_value=1, max_value=100)
samplerates = st.integers(min_value=1, max_value=96000)
fftsizes = st.integers(min_value=1, max_value=48000)


class TestMfcc(TestCase):
    @given(samplerates, fftsizes, frqs, frqs, filts)
    def test__mfcc(self, fps: int, n_fft: int, low: float, high: float, n_filters: int) -> None:
        assume(low<high)
        Mfcc(fps, n_fft, low, high, n_filters)
    
    #def test__mfcc_from_stft(self)
    #def test__mfcc_from_dft(self)
