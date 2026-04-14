#!/usr/bin/env python3

from functools import partial
import unittest
import numpy as np
from hypothesis import given
from hypothesis.strategies import composite, floats, integers, DrawFn

from apollon._defaults import SPL_REF
from apollon.signal import features
from apollon.signal import tools


frequencies = partial(floats, allow_nan=False, allow_infinity=False)

class TestAmp(unittest.TestCase):
    def test_amp_at_1Pa(self):
        sig = np.array([[1.0]], dtype=np.float64)
        res = tools.amp(features.spl(sig))
        self.assertTrue(np.allclose(res, sig))
        self.assertEqual(res.dtype.name, "float64")

    def test_amp_at_threshold(self):
        sig = np.array([[SPL_REF]], dtype=np.float64)
        res = tools.amp(features.spl(sig))
        self.assertTrue(np.allclose(res, sig))
        self.assertEqual(res.dtype.name, "float64")


class TestSinusoid(unittest.TestCase):
    def setUp(self):
        self.single_frq = 100
        self.multi_frq = (100, 200, 300)
        self.single_amp = .3
        self.multi_amp = (0.5, .3, .2)

    def test_returns_2darray_on_scalar_frq(self):
        sig = tools.sinusoid(self.single_frq)
        self.assertTrue(sig.ndim>1)


class TestAmpMod(unittest.TestCase):
    fps = 44100
    mod_idx = floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False, exclude_max=True)

    @composite
    @staticmethod
    def amp_mod_params(draw: DrawFn) -> tuple[float, float, float]:
        f_c = draw(frequencies(min_value=1, max_value=TestAmpMod.fps//2))
        f_m = draw(frequencies(min_value=0, max_value=f_c-1))
        m = draw(TestAmpMod.mod_idx)
        return (f_c, f_m, m)

    @given(amp_mod_params())
    def test_ampmod(self, params: tuple[float, float, float]) -> None:
        sig = tools.ampmod(*params)
        self.assertIsInstance(sig, np.ndarray)
        self.assertEqual(sig.dtype, np.float64)
        self.assertLessEqual(abs(sig).max(), 1)


class TestMelHzConverte(unittest.TestCase):
    @given(integers(min_value=0, max_value=1000000))
    def test(self, frq: int) -> None:
        res = tools.mel_to_hz(tools.hz_to_mel(frq))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.isclose(res, frq))


if __name__ == '__main__':
    unittest.main()
