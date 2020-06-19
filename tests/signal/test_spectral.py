import unittest
import numpy as np
import scipy as sp

from hypothesis import given
from hypothesis.strategies import integers, floats
from hypothesis.extra.numpy import arrays, array_shapes

from apollon.signal.spectral import fft, Dft, Stft, StftSegments
from apollon.signal.container import StftParams
from apollon.signal.tools import sinusoid


Array = np.ndarray

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



class TestStftSegmentsTimes(unittest.TestCase):
    def setUp(self):
        self.fps = 9000
        self.n_perseg = 512
        self.n_overlap = 256
        self.amps = np.array([1., .5, .25, .1, .05])
        self.frqs = np.array([440, 550, 660, 880, 1760])
        self.signal = sinusoid(self.frqs, self.amps, fps=self.fps)
        self.stft = StftSegments(self.fps)

    def times_extend_pad(self):
        cutter = Segmentation(self.n_perseg, self.n_overlap,
                              extend=True, pad=True)
        segs = cutter.transform(self.signal)
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


class TestSpectrum(unittest.TestCase):
    real_floats = floats(0, 1, allow_nan=False, allow_infinity=False)
    arr_2d_shapes = array_shapes(min_dims=2, max_dims=2,
                               min_side=1, max_side=100)
    float_2d_arrays = arrays(np.float, arr_2d_shapes,
                             elements=real_floats)

    @given(float_2d_arrays)
    def test_abs_is_real(self, inp: Array) -> None:
        dft = Dft(inp.shape[0], 'hamming', None)
        spctrm = dft.transform(inp)
        self.assertTrue(spctrm.abs.dtype.type is np.float64)

    @given(float_2d_arrays)
    def test_abs_ge_zero(self, inp: Array) -> None:
        dft = Dft(inp.shape[0], 'hamming', None)
        spctrm = dft.transform(inp)
        self.assertTrue(np.all(spctrm.abs>=0))

    @given(float_2d_arrays)
    def test_d_frq_is_positive_float(self, inp: Array) -> None:
        dft = Dft(inp.shape[0], 'hamming', None)
        spctrm = dft.transform(inp)
        dfrq = spctrm.d_frq
        self.assertTrue(isinstance(dfrq, float))
        self.assertTrue(dfrq>0)

    @given(float_2d_arrays)
    def test_frqs_is_positive_array(self, inp: Array) -> None:
        dft = Dft(inp.shape[0], 'hamming', None)
        spctrm = dft.transform(inp)
        frqs = spctrm.frqs
        self.assertTrue(isinstance(frqs, np.ndarray))
        self.assertTrue(frqs.dtype.type is np.float64)
        self.assertTrue(np.all(frqs>=0))

    @given(float_2d_arrays)
    def test_phase_within_pi(self, inp: Array) -> None:
        dft = Dft(inp.shape[0], 'hamming', None)
        spctrm = dft.transform(inp)
        phase = spctrm.phase
        self.assertTrue(phase.dtype.type is np.float64)
        self.assertTrue(np.all(-np.pi<=phase))
        self.assertTrue(np.all(phase<=np.pi))

    @given(float_2d_arrays)
    def test_power_is_positive_array(self, inp: Array) -> None:
        dft = Dft(inp.shape[0], 'hamming', None)
        spctrm = dft.transform(inp)
        power = spctrm.power
        self.assertTrue(power.dtype.type is np.float64)
        self.assertTrue(np.all(power>=0.0))

    @given(integers(min_value=1, max_value=10000))
    def test_n_fft(self, n_samples: int) -> None:
        sig = np.empty((n_samples, 1))
        dft = Dft(n_samples, 'hamming', None)
        y = dft.transform(sig)
        self.assertEqual(y._n_fft, sig.size)


class TestSpectrogram(unittest.TestCase):

    sp_args = {'window': 'hamming', 'nperseg': 512, 'noverlap': 256}
    ap_args = {'window': 'hamming', 'n_perseg': 512, 'n_overlap': 256}

    @given(integers(min_value=1000, max_value=20000))
    def test_times(self, fps) -> None:
        sig = np.random.rand(fps, 1)
        _, times, _ = sp.signal.stft(sig.squeeze(), fps,
                                     **TestSpectrogram.sp_args)
        stft = Stft(fps, **TestSpectrogram.ap_args)
        sxx = stft.transform(sig)
        self.assertTrue(np.allclose(times, sxx.times))

    @given(integers(min_value=2, max_value=44100))
    def test_frqs_and_bins_have_same_first_dim(self, nfft) -> None:
        fps = 9000
        sig = np.random.rand(fps, 1)
        stft = Stft(fps, **TestSpectrogram.ap_args)
        sxx = stft.transform(sig)
        self.assertEqual(sxx.frqs.shape[0], sxx.bins.shape[0])


if __name__ == '__main__':
    unittest.main()
