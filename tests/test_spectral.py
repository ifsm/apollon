#!/usr/bin/env python3

import unittest
import numpy as np

from apollon.signal.spectral import fft
from apollon.signal.tools import sinusoid


class TestFFT(unittest.TestCase):
    def setUp(self):
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.amps = np.array([1., .5, .25, .1, .05])
        self.signal_1d = sinusoid(self.frqs, self.amps, fps=9000)
        self.signal_2d = sinusoid(self.frqs, self.amps, fps=9000, retcomps=True).T

    def test_fft_1d(self):
        bins = np.absolute(fft(self.signal_1d))
        self.assertTrue(np.allclose(bins[self.frqs].T, self.amps))

    def test_fft_2d(self):
        bins = np.absolute(fft(self.signal_2d))
        idx = np.arange(self.frqs.size, dtype=int)
        self.assertTrue(np.allclose(bins[self.frqs, idx], self.amps))


if __name__ == '__main__':
    unittest.main()
