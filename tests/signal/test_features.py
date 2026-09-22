from typing import Callable
import unittest

import numpy as np

from hypothesis import given
from hypothesis import strategies as st
import hypothesis.extra.numpy as htn

from apollon.typing import FloatArray
from apollon.signal import features
from apollon.signal.spectral import Dft, Stft
from apollon.signal.tools import ampmod, sinusoid
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

    def test_scales_with_signal_level(self):
        """Loudness is finite, non-negative, and grows with signal level."""
        dft = Dft(fps=44100, window=None)
        quiet = features.loudness(dft.transform(sinusoid(440, fps=44100)))
        loud = features.loudness(dft.transform(sinusoid(440, 10, fps=44100)))
        self.assertTrue(np.all(np.isfinite(quiet)))
        self.assertTrue(np.all(quiet >= 0))
        self.assertGreater(loud.item(), quiet.item())


class TestSharpness(unittest.TestCase):

    def test_scales_with_signal_level(self):
        """Sharpness is finite and increases with signal level: masking
        spreads further upward in Bark at higher levels, so the same
        spectral shape reads sharper when louder."""
        dft = Dft(fps=44100, window=None)
        quiet = features.sharpness(dft.transform(sinusoid(440, fps=44100)))
        loud = features.sharpness(dft.transform(sinusoid(440, 10, fps=44100)))
        self.assertTrue(np.all(np.isfinite(quiet)))
        self.assertGreater(loud.item(), quiet.item())

    def test_din45692_reference_stimulus(self):
        """DIN 45692's calibration reference stimulus -- narrow-band noise
        from 920 Hz to 1080 Hz at 60 dB SPL -- must measure 1 acum.

        The masking slopes depend on absolute level, so this also checks
        that the stimulus is read at 60 dB SPL, whatever the window of the
        transform.
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

        for window in (None, 'hann'):
            with self.subTest(window=window):
                sxx = Dft(fps=fps, window=window).transform(band.reshape(-1, 1))
                sharp = features.sharpness(sxx)
                self.assertAlmostEqual(sharp.item(), 1.0, places=2)


class TestRoughness(unittest.TestCase):
    def setUp(self):
        self.fps = 44100
        self.sig = sinusoid(100, fps=self.fps, length=2)

    def _stft(self, frqs, n_perseg=4096):
        """Hann STFT of a sum of sinusoids, or of the signal ``frqs``.

        4096 samples resolve partials 33 Hz apart.
        """
        if not isinstance(frqs, np.ndarray):
            frqs = sinusoid(frqs, fps=self.fps)
        stft = Stft(fps=self.fps, n_perseg=n_perseg, n_overlap=n_perseg//2,
                    window='hann', extend=False, pad=False)
        return stft.transform(frqs)

    def _am_roughness(self, depth=1.0, n_perseg=4096):
        """Mean roughness of a 1 kHz tone amplitude-modulated at 70 Hz."""
        sig = ampmod(1000, 70, depth, 1.0, fps=self.fps, length=1.0)
        sxx = self._stft(sig, n_perseg)
        return features.roughness_helmholtz(sxx.d_frq, sxx.abs, 1500).mean()

    def test_single_array(self):
        dft = Dft(fps=self.fps, window=None)
        sxx = dft.transform(self.sig)
        features.roughness_helmholtz(sxx.d_frq, sxx.abs, frq_max=1000)

    def test_segmented_array(self):
        stft = Stft(fps=self.fps, n_perseg=2**8, n_overlap=2**7)
        sxx = stft.transform(self.sig)
        features.roughness_helmholtz(sxx.d_frq, sxx.abs, frq_max=1000, total=False)

    def test_follows_the_roughness_curve(self):
        """Two equal partials read the curve at their spacing, which peaks
        at 33.5 Hz."""
        spacings = (10, 20, 33, 50, 100)
        values = []
        for spacing in spacings:
            bins = np.zeros((301, 1))
            bins[[20, 20+spacing]] = 1.0
            values.append(features.roughness_helmholtz(1.0, bins, 200).item())
        curve = [frq/33.5 * np.exp(1 - frq/33.5) for frq in spacings]
        self.assertTrue(np.allclose(values, curve))
        self.assertEqual(int(np.argmax(values)), spacings.index(33))

    def test_pure_tone_is_smooth(self):
        """A single partial has no spacing, whatever its main lobe covers."""
        sxx = self._stft(1000)
        rough = features.roughness_helmholtz(sxx.d_frq, sxx.abs, 1500)
        self.assertTrue(np.allclose(rough, 0.0, rtol=0.0, atol=1e-12))

    def test_rises_continuously_with_modulation_depth(self):
        """No threshold makes the index jump or drop as the sidebands of an
        amplitude-modulated tone grow."""
        rough = [self._am_roughness(depth)
                 for depth in np.arange(0.05, 1.0001, 0.05)]
        steps = np.diff(rough)
        self.assertTrue(np.all(steps >= 0.0))
        self.assertLess(steps.max(), 0.2)

    def test_stable_across_frame_lengths(self):
        """Small changes in peak height between frame lengths move the index
        only a little."""
        rough = [self._am_roughness(n_perseg=n) for n in (2048, 4096, 8192)]
        self.assertLess(np.ptp(rough), 0.08)

    def test_rough_pair(self):
        """Partials 33 Hz apart are close to maximally rough."""
        sxx = self._stft((1000, 1033))
        rough = features.roughness_helmholtz(sxx.d_frq, sxx.abs, 1500)
        self.assertTrue(np.all(rough > 0.9))

    def test_input_is_not_modified(self):
        sxx = self._stft((1000, 1033))
        bins = sxx.abs
        before = bins.copy()
        features.roughness_helmholtz(sxx.d_frq, bins, 1500)
        self.assertTrue(np.array_equal(bins, before))

    def test_frq_max_beyond_spectrum_raises(self):
        bins = np.ones((101, 3))
        for inp, frq_max in ((bins, 101.0), (bins[:, 0], 50.0)):
            with self.subTest(shape=inp.shape, frq_max=frq_max):
                with self.assertRaises(ValueError):
                    features.roughness_helmholtz(1.0, inp, frq_max)

    def test_output_shapes(self):
        """One value per frame, or one row per spacing 0 ... frq_max."""
        bins = np.zeros((101, 3))
        self.assertEqual(features.roughness_helmholtz(2.0, bins, 100).shape,
                         (1, 3))
        self.assertEqual(features.roughness_helmholtz(2.0, bins, 100,
                                                      total=False).shape,
                         (51, 3))


if __name__ == '__main__':
    unittest.main()
