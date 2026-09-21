from unittest import TestCase

import numpy as np
import scipy.fft as spf
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from apollon.signal.cepstral import (ENERGY_FLOOR, MelCepstrogram, Mfcc,
                                     MfccSpectrogram, cepstral_coefs,
                                     log_mel_energies)
from apollon.signal.filter import mel_space, preemphasis, triangular_filter_bank
from apollon.signal.models import CepstrumParams, StftParams, TriangFilterSpec
from apollon.signal.spectral import Stft
from apollon.signal.tools import sinusoid


FPS = 9000
N_PERSEG = 512


def stft_params(**kwargs: object) -> StftParams:
    """Return the STFT params used throughout, overridden by ``kwargs``."""
    args = {"fps": FPS, "n_perseg": N_PERSEG, "n_overlap": N_PERSEG//2,
            "window": "hamming", "extend": True, "pad": True}
    args.update(kwargs)
    return StftParams(**args)    # type: ignore[arg-type]


class TestLogMelEnergies(TestCase):

    def setUp(self) -> None:
        self.frqs = np.fft.rfftfreq(N_PERSEG, 1/FPS)
        self.fbank = triangular_filter_bank(self.frqs, 80.0, 4000.0, 26)
        self.power = np.abs(np.random.rand(self.frqs.size, 17))

    def test_equals_db_of_band_energies(self) -> None:
        expected = 10 * np.log10(self.fbank @ self.power)
        self.assertTrue(np.allclose(log_mel_energies(self.power, self.fbank),
                                    expected))

    def test_shape_is_filters_by_segments(self) -> None:
        energies = log_mel_energies(self.power, self.fbank)
        self.assertEqual(energies.shape, (26, self.power.shape[1]))

    def test_floor_clamps_zeros(self) -> None:
        """A silent frame has no logarithm. It is floored, not ``-inf``."""
        energies = log_mel_energies(np.zeros_like(self.power), self.fbank)
        self.assertTrue(np.isfinite(energies).all())
        self.assertTrue(np.allclose(energies, 10*np.log10(ENERGY_FLOOR)))

    def test_non_positive_floor_raises(self) -> None:
        for floor in (0.0, -1e-10):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    log_mel_energies(self.power, self.fbank, floor)

    def test_axis_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            log_mel_energies(self.power[:-1], self.fbank)

    def test_top_db_is_off_by_default(self) -> None:
        self.assertTrue(np.array_equal(
            log_mel_energies(self.power, self.fbank),
            log_mel_energies(self.power, self.fbank, top_db=None)))

    def test_top_db_bounds_the_range_of_every_segment(self) -> None:
        plain = log_mel_energies(self.power, self.fbank)
        narrow = (plain.max(axis=0) - plain.min(axis=0)).min() / 2
        energies = log_mel_energies(self.power, self.fbank, top_db=narrow)
        spans = energies.max(axis=0) - energies.min(axis=0)
        self.assertTrue(np.allclose(spans, narrow))

    def test_top_db_wider_than_the_signal_is_a_no_op(self) -> None:
        plain = log_mel_energies(self.power, self.fbank)
        span = plain.max() - plain.min()
        self.assertTrue(np.array_equal(
            plain, log_mel_energies(self.power, self.fbank, top_db=2*span)))

    def test_top_db_floors_at_the_maximum_of_each_segment(self) -> None:
        """The reference is the loudest band of the segment itself."""
        plain = log_mel_energies(self.power, self.fbank)
        clamped = log_mel_energies(self.power, self.fbank, top_db=5.0)
        self.assertTrue(np.allclose(
            clamped, np.maximum(plain, plain.max(axis=0, keepdims=True) - 5.0)))

    def test_segments_are_independent(self) -> None:
        """A loud segment elsewhere leaves every other segment untouched.

        This is the point of the per-segment reference, and what the
        whole-spectrogram variant on ``topdb-global`` cannot offer.
        """
        louder = self.power.copy()
        louder[:, 0] *= 1e6
        a = log_mel_energies(self.power, self.fbank, top_db=5.0)[:, 1:]
        b = log_mel_energies(louder, self.fbank, top_db=5.0)[:, 1:]
        self.assertTrue(np.array_equal(a, b))

    def test_chunking_does_not_change_the_result(self) -> None:
        """Segment by segment equals all at once, which is what makes this
        variant usable with ``segment.FileSegmentation``.

        Agreement is to floating point rather than bit-exact: ``fbank @ power``
        takes a different BLAS path for a matrix than for a single column. The
        clamp itself contributes nothing to the difference.
        """
        whole = log_mel_energies(self.power, self.fbank, top_db=5.0)
        piecewise = np.hstack([
            log_mel_energies(self.power[:, i:i+1], self.fbank, top_db=5.0)
            for i in range(self.power.shape[1])])
        self.assertTrue(np.allclose(whole, piecewise, rtol=0, atol=1e-12))

    def test_differs_from_librosa_power_to_db(self) -> None:
        """Stated plainly: this variant is not librosa-compatible."""
        librosa = __import__("librosa")
        self.assertFalse(np.allclose(
            log_mel_energies(self.power, self.fbank, top_db=5.0),
            librosa.power_to_db(self.fbank @ self.power, top_db=5.0)))

    def test_negative_top_db_raises(self) -> None:
        with self.assertRaises(ValueError):
            log_mel_energies(self.power, self.fbank, top_db=-1.0)


class TestCepstralCoefs(TestCase):

    def setUp(self) -> None:
        self.energies = np.random.rand(26, 17)

    @given(st.integers(min_value=1, max_value=4))
    def test_matches_scipy_dct(self, dct_type: int) -> None:
        expected = spf.dct(self.energies, type=dct_type, axis=0, norm="ortho")
        coefs = cepstral_coefs(self.energies, dct_type)
        self.assertTrue(np.allclose(coefs, expected))

    @given(st.integers(min_value=1, max_value=4))
    def test_dct_is_unitary(self, dct_type: int) -> None:
        """``norm="ortho"`` conserves the energy of the band energies."""
        coefs = cepstral_coefs(self.energies, dct_type)
        self.assertAlmostEqual(float((coefs**2).sum()),
                               float((self.energies**2).sum()))

    @given(st.integers(min_value=1, max_value=4))
    def test_coefs_do_not_grow_with_the_number_of_bands(self,
                                                        dct_type: int) -> None:
        """The unnormalized DCT scaled with the band count; this one does not."""
        rng = np.random.default_rng(0)
        peaks = [np.absolute(cepstral_coefs(rng.random((n, 17)), dct_type)).max()
                 for n in (26, 104)]
        self.assertLess(max(peaks)/min(peaks), 2.0)

    def test_truncates_to_n_coefs(self) -> None:
        coefs = cepstral_coefs(self.energies, n_coefs=13)
        self.assertEqual(coefs.shape, (13, self.energies.shape[1]))

    def test_keeps_all_coefs_by_default(self) -> None:
        coefs = cepstral_coefs(self.energies)
        self.assertEqual(coefs.shape, self.energies.shape)

    def test_zero_lifter_gain_is_identity(self) -> None:
        plain = cepstral_coefs(self.energies, n_coefs=13)
        lifted = cepstral_coefs(self.energies, n_coefs=13, lifter_gain=24.0)
        self.assertTrue(np.allclose(cepstral_coefs(self.energies, n_coefs=13,
                                                   lifter_gain=0.0), plain))
        self.assertFalse(np.allclose(lifted, plain))

    def test_too_many_coefs_raises(self) -> None:
        with self.assertRaises(ValueError):
            cepstral_coefs(self.energies, n_coefs=self.energies.shape[0]+1)

    def test_non_positive_n_coefs_raises(self) -> None:
        with self.assertRaises(ValueError):
            cepstral_coefs(self.energies, n_coefs=0)

    def test_negative_lifter_gain_raises(self) -> None:
        with self.assertRaises(ValueError):
            cepstral_coefs(self.energies, lifter_gain=-1.0)


class TestMfcc(TestCase):

    def setUp(self) -> None:
        self.sig = sinusoid(440, fps=FPS)
        self.fb = TriangFilterSpec(low=80.0, high=4000.0, n_filters=26)
        self.mfcc = Mfcc(stft=stft_params(), fb=self.fb)

    def test_transform_returns_mel_cepstrogram(self) -> None:
        self.assertIsInstance(self.mfcc.transform(self.sig), MelCepstrogram)

    def test_shape_is_n_coefs_by_n_segments(self) -> None:
        sxx = Stft(fps=FPS, n_perseg=N_PERSEG, n_overlap=N_PERSEG//2,
                   window="hamming").transform(self.sig)
        res = self.mfcc.transform(self.sig)
        self.assertEqual(res.coefs.shape, (13, sxx.n_segments))
        self.assertEqual(res.n_coefs, 13)
        self.assertEqual(res.n_segments, sxx.n_segments)

    def test_coefs_are_finite(self) -> None:
        self.assertTrue(np.isfinite(self.mfcc.transform(self.sig).coefs).all())

    def test_energy_lands_in_band_of_sinusoid(self) -> None:
        """A 440 Hz sinusoid peaks in the mel band that holds 440 Hz."""
        res = self.mfcc.transform(self.sig)
        centers = mel_space(80.0, 4000.0, 28).ravel()[1:-1]
        peak = centers[res.log_mel_energies.mean(axis=1).argmax()]
        self.assertAlmostEqual(peak, 440.0, delta=60.0)

    def test_times_match_the_spectrogram(self) -> None:
        sxx = Stft(fps=FPS, n_perseg=N_PERSEG, n_overlap=N_PERSEG//2,
                   window="hamming").transform(self.sig)
        self.assertTrue(np.allclose(self.mfcc.transform(self.sig).times,
                                    sxx.times))

    def test_frqs_match_the_spectrogram(self) -> None:
        sxx = Stft(fps=FPS, n_perseg=N_PERSEG, n_overlap=N_PERSEG//2,
                   window="hamming").transform(self.sig)
        self.assertTrue(np.array_equal(self.mfcc.transform(self.sig).frqs,
                                       sxx.frqs))

    def test_filter_bank_is_built_once(self) -> None:
        first, second = (self.mfcc.transform(self.sig).filter_bank
                         for _ in range(2))
        self.assertIs(first, second)

    def test_params_are_reproducible(self) -> None:
        params = self.mfcc.transform(self.sig).params
        self.assertEqual(params, self.mfcc.params)
        self.assertEqual(type(params).model_validate_json(
            params.model_dump_json()), params)

    def test_stft_norm_is_forwarded(self) -> None:
        """``norm`` reaches the STFT instead of being dropped."""
        mfcc = Mfcc(stft=stft_params(norm=None), fb=self.fb)
        self.assertIsNone(mfcc.params.stft.norm)
        self.assertFalse(np.allclose(mfcc.transform(self.sig).coefs,
                                     self.mfcc.transform(self.sig).coefs))

    def test_top_db_reaches_the_energies(self) -> None:
        """``top_db`` flows from the constructor into the log energies."""
        mfcc = Mfcc(stft=stft_params(), fb=self.fb, top_db=20.0)
        res = mfcc.transform(self.sig)
        lme = res.log_mel_energies
        self.assertEqual(mfcc.params.top_db, 20.0)
        self.assertTrue(np.allclose(lme.max(axis=0) - lme.min(axis=0), 20.0))
        self.assertFalse(np.allclose(res.coefs,
                                     self.mfcc.transform(self.sig).coefs))

    def test_a_loud_event_leaves_other_frames_alone(self) -> None:
        """The property the whole-spectrogram variant cannot provide."""
        mfcc = Mfcc(stft=stft_params(), fb=self.fb, top_db=20.0)
        clicked = self.sig.copy()
        clicked[:64] = 10.0
        plain = mfcc.transform(self.sig).coefs
        loud = mfcc.transform(clicked).coefs
        self.assertTrue(np.allclose(plain[:, 20:], loud[:, 20:]))

    def test_top_db_defaults_to_off(self) -> None:
        self.assertIsNone(self.mfcc.params.top_db)

    def test_negative_top_db_raises_at_construction(self) -> None:
        with self.assertRaises(ValidationError):
            Mfcc(stft=stft_params(), fb=self.fb, top_db=-1.0)

    def test_stft_single_sided_is_forwarded(self) -> None:
        """``single_sided`` reaches the STFT instead of being dropped."""
        mfcc = Mfcc(stft=stft_params(single_sided=False), fb=self.fb)
        self.assertFalse(mfcc.params.stft.single_sided)
        self.assertFalse(np.allclose(mfcc.transform(self.sig).coefs,
                                     self.mfcc.transform(self.sig).coefs))

    def test_zero_preemphasis_leaves_signal_unchanged(self) -> None:
        plain = Mfcc(stft=stft_params(), fb=self.fb, preemphasis=0.0)
        sxx = Stft(fps=FPS, n_perseg=N_PERSEG, n_overlap=N_PERSEG//2,
                   window="hamming").transform(self.sig)
        self.assertTrue(np.allclose(
            plain.transform(self.sig).coefs,
            MfccSpectrogram(fb=self.fb).transform(sxx).coefs))

    def test_too_many_coefs_raises_at_construction(self) -> None:
        with self.assertRaises(ValidationError):
            Mfcc(stft=stft_params(),
                 fb=TriangFilterSpec(low=80.0, high=4000.0, n_filters=12),
                 cepstrum=CepstrumParams(n_coefs=13))

    def test_unresolvable_filter_bank_raises_at_construction(self) -> None:
        """Filters narrower than the frequency resolution cannot be built."""
        with self.assertRaises(ValueError):
            Mfcc(stft=stft_params(n_perseg=16, n_overlap=8),
                 fb=TriangFilterSpec(low=80.0, high=4000.0, n_filters=26))

    def test_cut_off_above_nyquist_raises(self) -> None:
        with self.assertRaises(ValueError):
            Mfcc(stft=stft_params(),
                 fb=TriangFilterSpec(low=80.0, high=FPS, n_filters=26))


class TestMfccSpectrogram(TestCase):

    def setUp(self) -> None:
        self.sig = sinusoid(440, fps=FPS)
        self.fb = TriangFilterSpec(low=80.0, high=4000.0, n_filters=26)
        self.stft = Stft(fps=FPS, n_perseg=N_PERSEG, n_overlap=N_PERSEG//2,
                         window="hamming")
        self.mfcc = MfccSpectrogram(fb=self.fb)

    def test_agrees_with_the_signal_path(self) -> None:
        """Pre-emphasizing by hand reproduces ``Mfcc`` exactly."""
        pre, _ = preemphasis(self.sig, 0.97)
        from_sxx = self.mfcc.transform(self.stft.transform(pre))
        from_sig = Mfcc(stft=stft_params(), fb=self.fb).transform(self.sig)
        self.assertTrue(np.allclose(from_sxx.coefs, from_sig.coefs))

    def test_preemphasis_is_unknown(self) -> None:
        """The transform cannot know what happened before the STFT."""
        res = self.mfcc.transform(self.stft.transform(self.sig))
        self.assertIsNone(res.params.preemphasis)

    def test_result_carries_the_spectrogram_params(self) -> None:
        sxx = self.stft.transform(self.sig)
        self.assertEqual(self.mfcc.transform(sxx).params.stft, sxx.params)

    def test_filter_bank_is_reused_across_calls(self) -> None:
        sxx = self.stft.transform(self.sig)
        self.assertIs(self.mfcc.transform(sxx).filter_bank,
                      self.mfcc.transform(sxx).filter_bank)

    def test_filter_bank_is_rebuilt_on_a_new_axis(self) -> None:
        wide = Stft(fps=FPS, n_perseg=2*N_PERSEG, n_overlap=N_PERSEG,
                    window="hamming")
        first = self.mfcc.transform(self.stft.transform(self.sig))
        second = self.mfcc.transform(wide.transform(self.sig))
        self.assertEqual(first.filter_bank.shape[1], first.frqs.shape[0])
        self.assertEqual(second.filter_bank.shape[1], second.frqs.shape[0])
        self.assertNotEqual(first.filter_bank.shape, second.filter_bank.shape)

    def test_params_hold_only_the_cepstral_stage(self) -> None:
        self.assertEqual(self.mfcc.params.fb, self.fb)
        self.assertFalse(hasattr(self.mfcc.params, "stft"))

    def test_too_many_coefs_raises_at_construction(self) -> None:
        with self.assertRaises(ValidationError):
            MfccSpectrogram(
                fb=TriangFilterSpec(low=80.0, high=4000.0, n_filters=12),
                cepstrum=CepstrumParams(n_coefs=13))


class TestMelCepstrogramAccess(TestCase):

    def setUp(self) -> None:
        self.res = Mfcc(stft=stft_params(),
                        fb=TriangFilterSpec(low=80.0, high=4000.0,
                                            n_filters=26)
                        ).transform(sinusoid(440, fps=FPS))

    def test_len_is_number_of_coefs(self) -> None:
        self.assertEqual(len(self.res), 13)

    def test_getitem_returns_one_coef_over_time(self) -> None:
        self.assertTrue(np.array_equal(self.res[0], self.res.coefs[0]))

    def test_repr_names_the_type(self) -> None:
        self.assertTrue(repr(self.res).startswith("MelCepstrogram("))
