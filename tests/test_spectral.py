import unittest
import numpy as np

from hypothesis import given
from hypothesis.strategies import integers

from apollon.signal import container
from apollon.signal import spectral
from apollon.signal.tools import sinusoid


class TestFft(unittest.TestCase):
    def setUp(self):
        self.fps = 9000
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.amps = np.array([1., .5, .25, .1, .05])
        self.signal = sinusoid(self.frqs, self.amps, fps=self.fps, comps=True)

    def test_input_shape(self):
        with self.assertRaises(ValueError):
            spectral.fft(np.random.randint(2, 100, (20, 20, 20)))

    def test_window_exists(self):
        with self.assertRaises(ValueError):
            spectral.fft(self.signal, window='whatever')

    @given(integers(min_value=1, max_value=44100))
    def test_nfft(self, n_fft):
        bins = spectral.fft(self.signal, n_fft=n_fft)
        self.assertEqual(bins.shape[0], n_fft//2+1)

    def test_transform(self):
        bins = np.absolute(spectral.fft(self.signal))
        idx = np.arange(self.frqs.size, dtype=int)
        self.assertTrue(np.allclose(bins[self.frqs, idx], self.amps))


class TestSpectrum(unittest.TestCase):
    def setUp(self):
        fps = 9000
        self.amps = np.array([1., .5, .25, .1, .05])
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.signal = sinusoid(self.frqs, self.amps, fps=fps)
        self.params = container.SpectrumParams(fps=fps)
        self.sxx = spectral.Spectrum(self.params)
        self.sxx.transform(self.signal)

    def test_abs(self):
        vals = self.sxx.abs
        self.assertTrue(isinstance(vals, np.ndarray))
        self.assertTrue(vals.dtype is np.dtype('float64'))
        self.assertTrue(np.all(vals > 0))

    def test_bins(self):
        vals = self.sxx.bins
        self.assertTrue(isinstance(vals, np.ndarray))
        self.assertTrue(vals.dtype is np.dtype('complex128'))

    def test_centroid(self):
        self.assertTrue(isinstance(self.sxx.centroid, float))

    def test_dfrq(self):
        vals = self.sxx.d_frq
        self.assertTrue(isinstance(vals, float))
        self.assertTrue(vals > 0)

    def test_frqs(self):
        vals = self.sxx.frqs
        self.assertTrue(isinstance(vals, np.ndarray))
        self.assertTrue(vals.dtype is np.dtype('float64'))
        self.assertTrue(np.all(vals>=0))

    def test_params(self):
        self.assertTrue(isinstance(self.sxx.params, container.SpectrumParams))

    def test_phase(self):
        vals = self.sxx.phase
        self.assertTrue(vals.dtype is np.dtype('float64'))
        self.assertTrue(np.all(vals <= np.pi))
        self.assertTrue(np.all(vals >= -np.pi))

    def test_power(self):
        vals = self.sxx.power
        self.assertTrue(vals.dtype is np.dtype('float64'))
        self.assertTrue(np.all(vals >= 0.))

    def test_transform(self):
        spc = spectral.Spectrum(self.params)
        spc.transform(self.signal)
        self.assertTrue(np.allclose(spc.abs[self.frqs].T, self.amps))

"""
    def test_spectrum_clip(self):
        spc = spectral.Spectrum(self.params)
        lcf, ucf, dbt = 560, 900, 20
        spc.clip(lcf, ucf, dbt)
        cond = np.logical_and(self.frqs > lcf, self.frqs <= ucf)
        frqs = self.frqs[cond]
        amps = self.amps[cond]
        self.assertTrue(np.allclose(spc.abs[frqs].T, self.amps))
"""

class TestSpectrogram(unittest.TestCase):
    def setUp(self):
        self.params = container.SpectrogramParams(fps=9000)

    def test_psectrogram_init(self):
        sxx= spectral.Spectrogram(self.params)




if __name__ == '__main__':
    unittest.main()
