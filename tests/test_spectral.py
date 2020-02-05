#!/usr/bin/env python3

import unittest
import numpy as np

from apollon.signal import container
from apollon.signal import spectral
from apollon.signal.tools import sinusoid


class TestFft(unittest.TestCase):
    def setUp(self):
        self.fps = 9000
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.amps = np.array([1., .5, .25, .1, .05])
        self.signal = sinusoid(self.frqs, self.amps, fps=self.fps)
        self.signal_comp = sinusoid(self.frqs, self.amps, fps=self.fps, comps=True)

    def test_fft_1d(self):
        bins = np.absolute(spectral.fft(self.signal))
        self.assertTrue(np.allclose(bins[self.frqs].T, self.amps))

    def test_fft_2d(self):
        bins = np.absolute(spectral.fft(self.signal_comp))
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
