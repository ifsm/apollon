#!/usr/bin/env python3

import unittest
import numpy as np

from apollon.container import FTParams
from apollon.signal.spectral import fft, Spectrum
from apollon.signal.tools import sinusoid


class TestSpectral(unittest.TestCase):
    def setUp(self):
        self.fps = 9000
        self.params = FTParams(self.fps)
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.amps = np.array([1., .5, .25, .1, .05])
        self.signal = sinusoid(self.frqs, self.amps, fps=self.fps)
        self.signal_comp = sinusoid(self.frqs, self.amps, fps=self.fps, retcomps=True).T

    def test_fft_1d(self):
        bins = np.absolute(fft(self.signal))
        self.assertTrue(np.allclose(bins[self.frqs].T, self.amps))

    def test_fft_2d(self):
        bins = np.absolute(fft(self.signal_comp))
        idx = np.arange(self.frqs.size, dtype=int)
        self.assertTrue(np.allclose(bins[self.frqs, idx], self.amps))

    def test_spectrum(self):
        spc = Spectrum(self.signal, self.params)
        self.assertTrue(np.allclose(spc.abs()[self.frqs].T, self.amps))

    def test_spectrum_clip(self):
        spc = Spectrum(self.signal, self.params)
        lcf, ucf, dbt = 560, 900, 20
        spc.clip(lcf, ucf, dbt)
        cond = np.logical_and(self.frqs > lcf, self.frqs <= ucf)
        frqs = self.frqs[cond]
        amps = self.amps[cond]
        self.assertTrue(np.allclose(spc.abs()[frqs].T, amps))

if __name__ == '__main__':
    unittest.main()
