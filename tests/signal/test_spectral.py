import unittest
import warnings

import numpy as np
import scipy as sp

from hypothesis import given
from hypothesis.strategies import integers, floats
from hypothesis.extra.numpy import arrays, array_shapes

from pydantic import ValidationError

from apollon.segment import ArraySegmentation
from apollon.segment.models import SegmentationParams
from apollon.signal.cepstral import _rfftfreq
from apollon.signal.models import StftParams
from apollon.signal.spectral import (fft, fft_length, full_scale_db, Dft, Stft,
                                     StftSegments)
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

    @given(integers(min_value=9000, max_value=44100))
    def test_nfft(self, n_fft):
        """``n_fft`` from the signal length (9000 samples) upwards."""
        bins = fft(self.signal, n_fft=n_fft)
        self.assertEqual(bins.shape[0], n_fft//2+1)

    def test_nfft_shorter_than_signal_raises(self):
        """A shorter FFT would silently crop the signal."""
        with self.assertRaises(ValueError):
            fft(self.signal, n_fft=self.signal.shape[0]-1)

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
                raw = fft(sig, n_fft=n_fft, norm=None, single_sided=False)
                ratio = (np.absolute(fft(sig, n_fft=n_fft))
                         / np.absolute(raw) * sig.shape[0])
                self.assertAlmostEqual(ratio[0, 0], 1.0)
                self.assertAlmostEqual(ratio[-1, 0], last)
                self.assertTrue(np.allclose(ratio[1:-1], 2.0))

    def test_norm_passes_convention_to_numpy(self):
        """Without the one-sided correction, the bins are numpy's own."""
        n = 512
        win = sp.signal.get_window('hamming', n).reshape(-1, 1)
        sig = np.random.default_rng(0).standard_normal((n, 1))
        for norm, expected in ((None, np.fft.rfft(sig*win, n, axis=0)),
                               ('ortho', np.fft.rfft(sig*win, n, axis=0,
                                                     norm='ortho')),
                               ('amplitude', np.fft.rfft(sig*win, n, axis=0)
                                / abs(win.sum()))):
            with self.subTest(norm=norm):
                bins = fft(sig, 'hamming', norm=norm, single_sided=False)
                self.assertTrue(np.array_equal(bins, expected))

    def test_single_sided_factor_follows_norm(self):
        """Amplitude conventions double, the unitary one scales by sqrt(2)."""
        n = 512
        sig = np.random.default_rng(0).standard_normal((n, 1))
        for norm, fac in ((None, 2.0), ('amplitude', 2.0),
                          ('ortho', np.sqrt(2.0))):
            with self.subTest(norm=norm):
                half = fft(sig, 'hamming', norm=norm, single_sided=False)
                full = fft(sig, 'hamming', norm=norm, single_sided=True)
                ratio = np.absolute(full) / np.absolute(half)
                self.assertAlmostEqual(ratio[0, 0], 1.0)
                self.assertAlmostEqual(ratio[-1, 0], 1.0)
                self.assertTrue(np.allclose(ratio[1:-1], fac))

    def test_ortho_conserves_energy_of_the_windowed_signal(self):
        """``ortho`` and ``single_sided`` together satisfy Parseval."""
        n = 512
        win = sp.signal.get_window('hamming', n).reshape(-1, 1)
        sig = np.random.default_rng(0).standard_normal((n, 1))
        for n_fft in (n, n+1):
            with self.subTest(n_fft=n_fft):
                bins = fft(sig, 'hamming', n_fft=n_fft, norm='ortho')
                self.assertAlmostEqual(float((np.absolute(bins)**2).sum()),
                                       float(((sig*win)**2).sum()))

    def test_amplitude_without_single_sided_reads_half(self):
        """The negative half keeps half of a real sinusoid's amplitude."""
        amp = 3.0
        sig = sinusoid(self.fps * 25 / 512, amp, fps=self.fps)[:512]
        bins = fft(sig, 'hamming', norm='amplitude', single_sided=False)
        self.assertAlmostEqual(np.absolute(bins).max(), amp/2)

    def test_unknown_norm_raises(self):
        for bad in ('backward', 'forward', True, False, 'whatever'):
            with self.subTest(norm=bad):
                with self.assertRaises(ValueError) as ctx:
                    fft(self.signal, norm=bad)
                self.assertIn('"amplitude"', str(ctx.exception))



class TestFftLength(unittest.TestCase):
    def test_unset_falls_back_to_the_frame_length(self):
        self.assertEqual(fft_length(None, 512), 512)

    def test_set_value_wins(self):
        self.assertEqual(fft_length(1024, 512), 1024)

    def test_transforms_and_helpers_agree(self):
        """Spectrogram, full_scale_db and the cepstral filter-bank axis
        resolve an unset ``n_fft`` alike."""
        unset = Stft(fps=9000, n_perseg=512, n_overlap=256)
        explicit = Stft(fps=9000, n_perseg=512, n_overlap=256, n_fft=512)
        sxx = unset.transform(np.zeros((9000, 1)))
        self.assertEqual(sxx.frqs.shape[0], 512//2 + 1)
        self.assertEqual(sxx.d_frq, 9000 / 512)
        self.assertTrue(np.array_equal(_rfftfreq(unset.params), sxx.frqs))
        self.assertEqual(full_scale_db(unset.params),
                         full_scale_db(explicit.params))


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
        self.assertIsNone(self._stft(norm=None).params.norm)
        self.assertIsNone(self._stft_segments(norm=None).params.norm)
        self.assertFalse(self._stft(single_sided=False).params.single_sided)

    def test_norm_defaults_to_amplitude(self) -> None:
        for params in (self._stft().params, self._stft_segments().params):
            self.assertEqual(params.norm, 'amplitude')
            self.assertTrue(params.single_sided)

    def test_bin_centered_sinusoid_has_unit_amplitude(self) -> None:
        """Unit amplitude on a bin center reads as 1.0 in the spectrum."""
        sxx = self._stft().transform(self.signal)
        self.assertAlmostEqual(sxx.abs.max(), 1.0, places=9)

    def test_raw_bins_scale_up_to_the_normed_ones(self) -> None:
        """Only the paired bins pick up the factor two."""
        normed = self._stft().transform(self.noise)
        raw = self._stft(norm=None, single_sided=False).transform(self.noise)
        self.assertTrue(np.allclose(raw.bins * self.factor, normed.bins))

    def test_segments_bin_centered_sinusoid_has_unit_amplitude(self) -> None:
        sxx = self._stft_segments().transform(self.segs)
        self.assertAlmostEqual(sxx.abs.max(), 1.0, places=9)

    def test_segments_raw_bins_scale_up_to_the_normed_ones(self) -> None:
        """Only the paired bins pick up the factor two."""
        normed = self._stft_segments().transform(self.noise_segs)
        raw = self._stft_segments(norm=None, single_sided=False).transform(
                self.noise_segs)
        self.assertTrue(np.allclose(raw.bins * self.factor, normed.bins))

    def test_agrees_with_stft_on_the_same_segmentation(self) -> None:
        """Both transforms normalize the same way."""
        for norm in (None, 'ortho', 'amplitude'):
            for single_sided in (True, False):
                with self.subTest(norm=norm, single_sided=single_sided):
                    kwargs = {'norm': norm, 'single_sided': single_sided}
                    from_sig = self._stft(**kwargs).transform(self.signal)
                    from_segs = self._stft_segments(**kwargs).transform(
                            self.segs)
                    self.assertTrue(np.array_equal(from_sig.bins,
                                                   from_segs.bins))

    def test_params_round_trip(self) -> None:
        """Every setting survives serialization."""
        for norm in (None, 'ortho', 'amplitude'):
            with self.subTest(norm=norm):
                params = self._stft(norm=norm, single_sided=False).params
                self.assertEqual(
                        type(params).model_validate_json(
                            params.model_dump_json()), params)

    def test_legacy_boolean_norm_is_rejected(self) -> None:
        """The boolean flag is gone; old params must not load silently."""
        with self.assertRaises(ValidationError):
            StftParams.model_validate_json(
                    '{"fps": 9000, "norm": true, "n_perseg": 512,'
                    ' "n_overlap": 256, "extend": true, "pad": true}')


class TestStftSegments(unittest.TestCase):
    fps = 1000

    def setUp(self):
        self.seg_params = SegmentationParams(n_perseg=64, n_overlap=32)
        self.sig = np.random.default_rng(0).normal(size=(self.fps, 1))

    def _segments(self, n_overlap):
        return ArraySegmentation(64, n_overlap).transform(self.sig)

    def test_transforms_matching_segments(self):
        sxx = StftSegments(self.fps, self.seg_params).transform(
                self._segments(32))
        self.assertEqual(sxx.n_segments, self._segments(32).n_segs)

    def test_rejects_other_segmentation(self):
        """A mismatch would give the spectrogram a wrong time axis."""
        with self.assertRaises(ValueError):
            StftSegments(self.fps, self.seg_params).transform(
                    self._segments(48))

    def test_uses_no_deprecated_api(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            StftSegments(self.fps, self.seg_params)


class TestFullScaleDb(unittest.TestCase):
    """``full_scale_db`` must agree with what ``fft`` actually produces."""

    fps = 9000
    n_perseg = 512

    def _peak_db(self, **kwargs) -> tuple[float, float]:
        stft = Stft(fps=self.fps, n_perseg=self.n_perseg,
                    n_overlap=self.n_perseg//2, extend=False, pad=False,
                    **kwargs)
        n_fft = kwargs.get('n_fft') or self.n_perseg
        # a unit sinusoid on bin 50 of the n_fft grid, far from DC and Nyquist
        sig = sinusoid(self.fps * 50 / n_fft, 1.0, fps=self.fps)
        peak = 10 * np.log10(stft.transform(sig).power.max())
        return peak, full_scale_db(stft.params)

    def test_matches_a_full_scale_sinusoid(self) -> None:
        for window in (None, 'hamming', 'hann'):
            for norm in (None, 'ortho', 'amplitude'):
                for single_sided in (True, False):
                    with self.subTest(window=window, norm=norm,
                                      single_sided=single_sided):
                        peak, expected = self._peak_db(
                            window=window, norm=norm,
                            single_sided=single_sided)
                        self.assertAlmostEqual(peak, expected, places=9)

    def test_accounts_for_zero_padding(self) -> None:
        """``'ortho'`` divides by the FFT length, not the window length."""
        for norm in (None, 'ortho', 'amplitude'):
            with self.subTest(norm=norm):
                peak, expected = self._peak_db(window='hann', norm=norm,
                                               n_fft=2*self.n_perseg)
                self.assertAlmostEqual(peak, expected, places=9)

    def test_is_zero_under_the_default_scaling(self) -> None:
        """The default scaling is calibrated to full scale already."""
        params = Stft(fps=self.fps, n_perseg=self.n_perseg,
                      n_overlap=self.n_perseg//2, window='hamming').params
        self.assertEqual(full_scale_db(params), 0.0)


class TestMsPower(unittest.TestCase):
    """``ms_power`` sums to the mean square of the signal, whatever the
    scaling of the transform."""

    fps = 9000
    n_perseg = 512

    def test_dft_satisfies_parseval_under_any_scaling(self) -> None:
        """A spectrum sums to the window-weighted mean square of its input."""
        n = 1000
        sig = np.random.default_rng(0).standard_normal((n, 1))
        for window in (None, 'hann', 'blackman'):
            win = sp.signal.get_window(window or 'rect', n).reshape(-1, 1)
            expected = float(((sig*win)**2).sum() / (win**2).sum())
            for norm in (None, 'ortho', 'amplitude'):
                for single_sided in (True, False):
                    for n_fft in (None, n+1, 2048):
                        with self.subTest(window=window, norm=norm,
                                          single_sided=single_sided,
                                          n_fft=n_fft):
                            dft = Dft(self.fps, window, n_fft, norm=norm,
                                      single_sided=single_sided)
                            total = dft.transform(sig).ms_power.sum()
                            self.assertAlmostEqual(float(total), expected,
                                                   places=9)

    def test_stft_frames_hold_the_mean_square_of_a_sinusoid(self) -> None:
        """A sinusoid of amplitude A has mean square A²/2 in every frame."""
        amp = 3.0
        sig = sinusoid(self.fps * 25 / self.n_perseg, amp, fps=self.fps)
        for window in (None, 'hann'):
            for norm in (None, 'ortho', 'amplitude'):
                with self.subTest(window=window, norm=norm):
                    stft = Stft(fps=self.fps, n_perseg=self.n_perseg,
                                n_overlap=self.n_perseg//2, window=window,
                                norm=norm, extend=False, pad=False)
                    totals = stft.transform(sig).ms_power.sum(axis=0)
                    self.assertTrue(np.allclose(totals, amp**2 / 2))


if __name__ == '__main__':
    unittest.main()
