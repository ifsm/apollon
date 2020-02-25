import unittest
import numpy as np
from scipy.signal import stft

from hypothesis import given
from hypothesis.strategies import integers

from apollon.signal.spectral import fft, Dft, StftSegments
from apollon.signal.container import STParams
from apollon.signal.tools import sinusoid


class TestFft(unittest.TestCase):
    def setUp(self):
        self.fps = 9000
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.amps = np.array([1., .5, .25, .1, .05])
        self.signal = sinusoid(self.frqs, self.amps, fps=self.fps, comps=True)

    def test_input_shape(self):
        with self.assertRaises(ValueError):
            fft(np.random.randint(2, 100, (20, 20, 20)))

    def test_window_exists(self):
        with self.assertRaises(ValueError):
            fft(self.signal, window='whatever')

    @given(integers(min_value=1, max_value=44100))
    def test_nfft(self, n_fft):
        bins = fft(self.signal, n_fft=n_fft)
        self.assertEqual(bins.shape[0], n_fft//2+1)

    def test_transform(self):
        bins = np.absolute(fft(self.signal))
        idx = np.arange(self.frqs.size, dtype=int)
        self.assertTrue(np.allclose(bins[self.frqs, idx], self.amps))


class TestDft(unittest.TestCase):
    def setUp(self):
        self.fps = 9000
        self.amps = np.array([1., .5, .25, .1, .05])
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.signal = sinusoid(self.frqs, self.amps, fps=self.fps)
        self.dft = Dft(self.fps)
        self.sxx = self.dft.transform(self.signal)

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
        self.assertTrue(isinstance(self.sxx.params, STParams))

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
        dft = Dft(self.fps)
        spc = dft.transform(self.signal)
        self.assertTrue(np.allclose(spc.abs[self.frqs].T, self.amps))


class TestStftSegmentsTimes(unittest.TestCase):
    def setUp(self):
        self.fps = 9000
        self.n_perseg = 512
        self.n_overlap = 256
        self.amps = np.array([1., .5, .25, .1, .05])
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.signal = sinusoid(self.frqs, self.amps, fps=fps)
        self.params = StftParams(fps=fps)
        self.stft = StftSegments(self.params)

    def times_extend_pad(self):
        segmenter = Segmentation(self.n_perseg, self.n_overlap,
                                 extend=True, pad=True)
        segs = segmenter.transform(self.signal)
        sxx = self.stft.transform(segs)
        frqs, times, bins = stft(self.signal.squeezs(), self.fps, 'hamming',
                                 self.n_perseg, self.n_overlap,
                                 boundary='zeros', padded=True)
        self.assertEqual(sxx.times.size, times.size)
        self.assertTrue(np.allclose(sxx.times.squeeze(), times))

    def times_extend_no_pad(self):
        segmenter = Segmentation(self.n_perseg, self.n_overlap,
                                 extend=True, pad=False)
        segs = segmenter.transform(self.signal)
        sxx = self.stft.transform(segs)
        frqs, times, bins = stft(self.signal.squeezs(), self.fps, 'hamming',
                                 self.n_perseg, self.n_overlap,
                                 boundary='zeros', padded=False)
        self.assertEqual(sxx.times.size, times.size)
        self.assertTrue(np.allclose(sxx.times.squeeze(), times))

    def times_no_extend_pad(self):
        segmenter = Segmentation(self.n_perseg, self.n_overlap,
                                 extend=False, pad=True)
        segs = segmenter.transform(self.signal)
        sxx = self.stft.transform(segs)
        frqs, times, bins = stft(self.signal.squeezs(), self.fps, 'hamming',
                                 self.n_perseg, self.n_overlap,
                                 boundary=None, padded=True)
        self.assertEqual(sxx.times.size, times.size)
        self.assertTrue(np.allclose(sxx.times.squeeze(), times))

    def times_no_extend_no_pad(self):
        segmenter = Segmentation(self.n_perseg, self.n_overlap,
                                 extend=False, pad=False)
        segs = segmenter.transform(self.signal)
        sxx = self.stft.transform(segs)
        frqs, times, bins = stft(self.signal.squeezs(), self.fps, 'hamming',
                                 self.n_perseg, self.n_overlap,
                                 boundary=None, padded=False)
        self.assertEqual(sxx.times.size, times.size)
        self.assertTrue(np.allclose(sxx.times.squeeze(), times))


if __name__ == '__main__':
    unittest.main()
