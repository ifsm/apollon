import unittest
import numpy as np
import scipy as sp

from hypothesis import given
from hypothesis.strategies import integers, floats
from hypothesis.extra.numpy import arrays, array_shapes

from apollon.segment import ArraySegmentation
from apollon.segment.models import SegmentationParams
from apollon.signal.spectral import fft, Dft, Stft, StftSegments
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

    def test_dc_is_not_doubled(self):
        """The zeroth bin has no partner in the negative half spectrum."""
        amp = 3.0
        bins = fft(np.full((512, 1), amp))
        self.assertAlmostEqual(np.absolute(bins)[0, 0], amp)

    def test_nyquist_is_not_doubled(self):
        """For even ``n_fft`` the last bin is Nyquist, which is unpaired."""
        amp = 3.0
        sig = (amp * (-1.0)**np.arange(512)).reshape(-1, 1)
        self.assertAlmostEqual(np.absolute(fft(sig))[-1, 0], amp)

    def test_last_bin_is_doubled_for_odd_n_fft(self):
        """Odd ``n_fft`` has no Nyquist bin, so its last bin is paired."""
        n_fft = 513
        amp = 3.0
        frq = self.fps * (n_fft//2) / n_fft
        sig = sinusoid(frq, amp, fps=self.fps)[:n_fft]
        self.assertAlmostEqual(np.absolute(fft(sig, n_fft=n_fft))[-1, 0], amp)

    def test_only_paired_bins_are_doubled(self):
        """Parity of ``n_fft`` decides, not the length of the signal."""
        sig = np.random.default_rng(0).standard_normal((512, 1))
        for n_fft, last in ((512, 1.0), (513, 2.0)):
            with self.subTest(n_fft=n_fft):
                # the rect window sums to the signal length
                ratio = (np.absolute(fft(sig, n_fft=n_fft))
                         / np.absolute(fft(sig, n_fft=n_fft, norm=False))
                         * sig.shape[0])
                self.assertAlmostEqual(ratio[0, 0], 1.0)
                self.assertAlmostEqual(ratio[-1, 0], last)
                self.assertTrue(np.allclose(ratio[1:-1], 2.0))



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
    float_2d_arrays = arrays(float, arr_2d_shapes,
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


class TestStftNorm(unittest.TestCase):
    """``norm`` scales the bins such that unit amplitude reads as unit
    amplitude in the spectrum."""

    fps = 9000
    n_perseg = 512
    window = 'hamming'

    def setUp(self):
        self.seg_params = SegmentationParams(n_perseg=self.n_perseg,
                                             n_overlap=self.n_perseg//2,
                                             extend=False, pad=False)
        frq = self.fps * 25 / self.n_perseg
        self.signal = sinusoid(frq, fps=self.fps)
        self.segs = ArraySegmentation(**self.seg_params.model_dump()).transform(
                self.signal)
        # broadband, so that the DC and Nyquist bins carry something
        self.noise = np.random.default_rng(0).standard_normal((self.fps, 1))
        self.noise_segs = ArraySegmentation(
                **self.seg_params.model_dump()).transform(self.noise)
        win_sum = sp.signal.get_window(self.window, self.n_perseg).sum()
        # ``n_perseg`` is even, so the last bin is Nyquist and stays unpaired
        self.factor = np.full((self.n_perseg//2 + 1, 1), 2.0 / win_sum)
        self.factor[[0, -1]] = 1.0 / win_sum

    def _stft(self, **kwargs) -> Stft:
        return Stft(fps=self.fps, n_perseg=self.n_perseg,
                    n_overlap=self.n_perseg//2, window=self.window,
                    extend=False, pad=False, **kwargs)

    def _stft_segments(self, **kwargs) -> StftSegments:
        return StftSegments(fps=self.fps, seg_params=self.seg_params,
                            window=self.window, **kwargs)

    def test_norm_is_stored_in_params(self) -> None:
        self.assertFalse(self._stft(norm=False).params.norm)
        self.assertFalse(self._stft_segments(norm=False).params.norm)

    def test_norm_defaults_to_true(self) -> None:
        self.assertTrue(self._stft().params.norm)
        self.assertTrue(self._stft_segments().params.norm)

    def test_bin_centered_sinusoid_has_unit_amplitude(self) -> None:
        """Unit amplitude on a bin center reads as 1.0 in the spectrum."""
        sxx = self._stft().transform(self.signal)
        self.assertAlmostEqual(sxx.abs.max(), 1.0, places=9)

    def test_norm_false_leaves_bins_unscaled(self) -> None:
        """Only the paired bins pick up the factor two."""
        normed = self._stft(norm=True).transform(self.noise)
        raw = self._stft(norm=False).transform(self.noise)
        self.assertTrue(np.allclose(raw.bins * self.factor, normed.bins))

    def test_segments_bin_centered_sinusoid_has_unit_amplitude(self) -> None:
        sxx = self._stft_segments().transform(self.segs)
        self.assertAlmostEqual(sxx.abs.max(), 1.0, places=9)

    def test_segments_norm_false_leaves_bins_unscaled(self) -> None:
        """Only the paired bins pick up the factor two."""
        normed = self._stft_segments(norm=True).transform(self.noise_segs)
        raw = self._stft_segments(norm=False).transform(self.noise_segs)
        self.assertTrue(np.allclose(raw.bins * self.factor, normed.bins))

    def test_agrees_with_stft_on_the_same_segmentation(self) -> None:
        """Both transforms normalize the same way."""
        for norm in (True, False):
            with self.subTest(norm=norm):
                from_sig = self._stft(norm=norm).transform(self.signal)
                from_segs = self._stft_segments(norm=norm).transform(self.segs)
                self.assertTrue(np.array_equal(from_sig.bins, from_segs.bins))


if __name__ == '__main__':
    unittest.main()
