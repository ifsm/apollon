from typing import Callable
import unittest

import numpy as np

from hypothesis import given
from hypothesis import strategies as st
import hypothesis.extra.numpy as htn

from apollon.typing import FloatArray
from apollon.signal import features
from apollon.signal.spectral import Dft, Stft
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



class TestLoudness(unittest.TestCase):

    def test_scales_with_bin_magnitude(self):
        """Loudness is finite, non-negative, and grows with signal level."""
        dft = Dft(fps=44100, window=None)
        sxx = dft.transform(sinusoid(440, fps=44100))
        quiet = features.loudness(sxx.frqs, sxx.bins)
        loud = features.loudness(sxx.frqs, sxx.bins * 10)
        self.assertTrue(np.all(np.isfinite(quiet)))
        self.assertTrue(np.all(quiet >= 0))
        self.assertGreater(loud.item(), quiet.item())


class TestSharpness(unittest.TestCase):

    def test_scales_with_bin_magnitude(self):
        """Sharpness is finite and unaffected by uniform level scaling."""
        dft = Dft(fps=44100, window=None)
        sxx = dft.transform(sinusoid(440, fps=44100))
        quiet = features.sharpness(sxx.frqs, sxx.bins)
        loud = features.sharpness(sxx.frqs, sxx.bins * 10)
        self.assertTrue(np.all(np.isfinite(quiet)))
        self.assertAlmostEqual(quiet.item(), loud.item())

    @unittest.expectedFailure
    def test_din45692_reference_stimulus(self):
        """DIN 45692's calibration reference stimulus -- narrow-band noise
        from 920 Hz to 1080 Hz at 60 dB SPL -- must measure 1 acum.

        Tracked as open issue #11 (see
        analyze-src-apollon-signal-critical-band-elegant-teacup.md): apollon's
        filter_bank hard-assigns each FFT bin to exactly one integer Bark
        band with no excitation spreading (auditory-filter leakage into
        neighbouring bands), so the energy-weighted Bark centroid sits below
        the ~9.09 Bark the 0.11 DIN constant assumes. Currently measures
        ~0.93 acum. Remove the ``expectedFailure`` marker once spreading is
        implemented.
        """
        fps = 44100
        n = fps * 2
        rng = np.random.default_rng(0)
        white = rng.normal(size=n)
        spec = np.fft.rfft(white)
        frq_axis = np.fft.rfftfreq(n, 1/fps)
        spec[~((frq_axis >= 920) & (frq_axis <= 1080))] = 0
        band = np.fft.irfft(spec, n)
        target_rms = SPL_REF * 10**(60/20)
        band *= target_rms / np.sqrt(np.mean(band**2))

        dft = Dft(fps=fps, window=None)
        sxx = dft.transform(band.reshape(-1, 1))
        sharp = features.sharpness(sxx.frqs, sxx.bins)
        self.assertAlmostEqual(sharp.item(), 1.0, places=2)


class TestRoughness(unittest.TestCase):
    def setUp(self):
        self.fps = 44100
        self.sig = sinusoid(100, length=2)

    def test_single_array(self):
        dft = Dft(fps=self.fps, window=None)
        sxx = dft.transform(self.sig)
        features.roughness_helmholtz(sxx.d_frq, sxx.abs, frq_max=1000)

    def test_segmented_array(self):
        stft = Stft(fps=self.fps, n_perseg=2**8, n_overlap=2**7)
        sxx = stft.transform(self.sig)
        features.roughness_helmholtz(sxx.d_frq, sxx.abs, frq_max=1000, total=False)


if __name__ == '__main__':
    unittest.main()
