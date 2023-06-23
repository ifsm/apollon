from typing import Callable
import unittest

import numpy as np

from hypothesis import given, assume
from hypothesis import strategies as st
import hypothesis.extra.numpy as htn

from apollon.types import Array, FloatArray
from apollon.signal import features
from apollon.signal.spectral import Dft
from apollon.signal.tools import sinusoid
from apollon._defaults import SPL_REF

finite_float_arrays = htn.arrays(
        np.float64,
        htn.array_shapes(min_dims=2, max_dims=2, min_side=2),
        elements=st.floats(min_value=-1000.0, max_value=1000.0,
                           allow_nan=False, allow_infinity=False))

sample_rates = st.integers(min_value=4, max_value=100000)

@st.composite
def rates_and_frequencies(draw: Callable, elements: st.SearchStrategy = sample_rates
                          ) -> tuple[int, float]:
    fps = draw(elements)
    frq = draw(st.integers(min_value=1, max_value=fps//2-1))
    return fps, frq

'''
class TestCdim(unittest.TestCase):
    def setUp(self):
        self.data = sinusoid((300, 600), [.2, .1], fps=3000, noise=None)
        self.ecr = features.cdim(self.data, delay=14, m_dim=80, n_bins=1000,
                scaling_size=10, mode='bader')

    def test_cdim_returns_array(self):
        self.assertTrue(isinstance(self.ecr, Array))

    def test_cdim_gt_zero(self):
        self.assertTrue(np.all(self.ecr > 0))
'''

class TestEnergy(unittest.TestCase):
    @given(finite_float_arrays)
    def test_energy_positive(self, sig: FloatArray) -> None:
        res = features.energy(sig)
        cond = res >= 0
        self.assertTrue(cond.all())
        self.assertEqual(res.ndim, 2)
        self.assertTrue(res.shape, (1, sig.shape[1]))


class TestSpl(unittest.TestCase):

    def test_spl_at_1Pa(self) -> None:
        sig = np.array([[1.0]], dtype=np.float64)
        res = features.spl(sig)
        self.assertGreater(res, 93.9)
        self.assertLessEqual(res, 94.0)

    def test_spl_at_threshold(self) -> None:
        sig = np.array([[SPL_REF]], dtype=np.float64)
        res = features.spl(sig)
        self.assertEqual(res, 0.0)


class TestRms(unittest.TestCase):

    @given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=2))
    def test_rms(self, dx: int, dy: int) -> None:
        sig = np.ones((dx, dy), dtype=np.float64)
        res = features.rms(sig)
        self.assertTrue(np.array_equal(res, np.ones((1, dy))))


class TestSpectralCentroid(unittest.TestCase):

    @given(rates_and_frequencies())
    def test_centroid(self, params):
        fps, frq = params
        sig = sinusoid(frq, fps=fps)
        dft = Dft(fps=fps)
        sxx = dft.transform(sig)
        spc = features.spectral_centroid(sxx.frqs, sxx.power)
        self.assertAlmostEqual(spc.item(), frq)


class TestSpectralSpread(unittest.TestCase):
   @given(rates_and_frequencies())
   def test_spread(self, params):
       fps, frq = params
       sig = sinusoid(frq, fps=fps)
       dft = Dft(fps=fps, window=None)
       sxx = dft.transform(sig)
       sps = features.spectral_spread(sxx.frqs, sxx.power)
       self.assertLess(sps.item(), 1.0)

   @given(rates_and_frequencies())
   def test_spread(self, params):
       fps, frq = params
       sig = sinusoid(frq, fps=fps)
       dft = Dft(fps=fps, window=None)
       sxx = dft.transform(sig)
       spc = features.spectral_centroid(sxx.frqs, sxx.power)
       sps = features.spectral_spread(sxx.frqs, sxx.power)
       sps_wc = features.spectral_spread(sxx.frqs, sxx.power, spc)
       self.assertEqual(sps.item(), sps_wc.item())
       self.assertLess(sps.item(), 1.0)


if __name__ == '__main__':
    unittest.main()
