#!/usr/bin/env python3

from functools import partial
from math import ceil
import unittest
import numpy as np
from hypothesis import given
from hypothesis.strategies import (composite, floats, integers, sampled_from,
                                   DrawFn)

from apollon._defaults import SPL_REF
from apollon.signal import features
from apollon.signal import tools
from apollon.typing import FloatArray


frequencies = partial(floats, allow_nan=False, allow_infinity=False)

TrimSpec = tuple[FloatArray, int, int | None, int | None]

class TestAmp(unittest.TestCase):
    def test_amp_at_1Pa(self):
        sig = np.array([[1.0]], dtype=np.float64)
        res = tools.amp(features.spl(sig))
        self.assertTrue(np.allclose(res, sig))
        self.assertEqual(res.dtype.name, "float64")

    def test_amp_at_threshold(self):
        sig = np.array([[SPL_REF]], dtype=np.float64)
        res = tools.amp(features.spl(sig))
        self.assertTrue(np.allclose(res, sig))
        self.assertEqual(res.dtype.name, "float64")


class TestCorrCoefPearson(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x_sig = rng.normal(size=200)
        self.y_sig = 0.5 * self.x_sig + rng.normal(size=200)

    def test_matches_numpy(self):
        res = tools.corr_coef_pearson(self.x_sig, self.y_sig)
        self.assertAlmostEqual(res, np.corrcoef(self.x_sig, self.y_sig)[0, 1])

    def test_is_invariant_to_offset_and_scale(self):
        res = tools.corr_coef_pearson(self.x_sig, 3*self.y_sig + 2)
        self.assertAlmostEqual(res, tools.corr_coef_pearson(self.x_sig,
                                                            self.y_sig))

    def test_reaches_the_bounds(self):
        self.assertAlmostEqual(
            tools.corr_coef_pearson(self.x_sig, 2*self.x_sig), 1.0)
        self.assertAlmostEqual(
            tools.corr_coef_pearson(self.x_sig, -self.x_sig), -1.0)

    def test_acf_pearson_matches_numpy_per_lag(self):
        res = tools.acf_pearson(self.x_sig)
        expected = [1.0] + [np.corrcoef(self.x_sig[:-lag],
                                        self.x_sig[lag:])[0, 1]
                            for lag in range(1, self.x_sig.size-1)]
        self.assertTrue(np.allclose(res, expected))


class TestAcf(unittest.TestCase):
    def test_matches_lagged_products(self):
        sig = np.random.default_rng(0).normal(size=300)
        expected = [sig[:sig.size-lag] @ sig[lag:] / (sig @ sig)
                    for lag in range(sig.size)]
        self.assertTrue(np.allclose(tools.acf(sig), expected))

    def test_silent_input(self):
        expected = np.zeros(10)
        expected[0] = 1.0
        self.assertTrue(np.array_equal(tools.acf(np.zeros(10)), expected))


class TestCWeighting(unittest.TestCase):
    def test_is_zero_db_at_1khz(self):
        self.assertAlmostEqual(tools.c_weighting(np.array([1000.0])).item(),
                               1.0)

    def test_matches_the_iec_table(self):
        """IEC 61672-1 C-weighting in dB, to 0.1 dB."""
        frqs = np.array([31.5, 63.0, 125.0, 4000.0, 8000.0])
        expected = np.array([-3.0, -0.8, -0.2, -0.8, -3.0])
        res = 20 * np.log10(tools.c_weighting(frqs))
        self.assertTrue(np.allclose(res, expected, atol=0.1))


class TestNormalize(unittest.TestCase):
    def test_scales_each_channel_to_unit_peak(self):
        sig = np.array([[0.5, -2.0], [-0.25, 1.0]])
        res = tools.normalize(sig)
        self.assertTrue(np.allclose(np.abs(res).max(axis=0), 1.0))
        self.assertTrue(np.allclose(res, [[1.0, -1.0], [-0.5, 0.5]]))

    def test_silent_channel_stays_zero(self):
        sig = np.zeros((10, 2))
        sig[:, 1] = np.linspace(-2.0, 1.0, 10)
        with np.errstate(all='raise'):
            res = tools.normalize(sig)
        self.assertTrue(np.all(res[:, 0] == 0.0))
        self.assertAlmostEqual(np.abs(res[:, 1]).max(), 1.0)


class TestLimit(unittest.TestCase):
    def setUp(self):
        self.inp = tools.amp([20.0, 40.0, 60.0])

    def test_without_boundaries_returns_a_copy(self):
        res = tools.limit(self.inp)
        self.assertTrue(np.array_equal(res, self.inp))
        self.assertFalse(np.shares_memory(res, self.inp))

    def test_lower_boundary_raises_values(self):
        res = tools.limit(self.inp, ldb=40.0)
        self.assertTrue(np.allclose(res, tools.amp([40.0, 40.0, 60.0])))

    def test_upper_boundary_lowers_values(self):
        res = tools.limit(self.inp, udb=40.0)
        self.assertTrue(np.allclose(res, tools.amp([20.0, 40.0, 40.0])))

    def test_both_boundaries(self):
        res = tools.limit(self.inp, ldb=30.0, udb=50.0)
        self.assertTrue(np.allclose(res, tools.amp([30.0, 40.0, 50.0])))

    def test_accepts_numpy_scalars(self):
        res = tools.limit(self.inp, ldb=np.float32(40.0))
        self.assertTrue(np.allclose(res, tools.amp([40.0, 40.0, 60.0])))

    def test_inverted_boundaries_raise(self):
        with self.assertRaises(ValueError):
            tools.limit(self.inp, ldb=50.0, udb=30.0)


class TestSinusoid(unittest.TestCase):
    def setUp(self):
        self.single_frq = 100
        self.multi_frq = (100, 200, 300)
        self.single_amp = .3
        self.multi_amp = (0.5, .3, .2)

    def test_returns_2darray_on_scalar_frq(self):
        sig = tools.sinusoid(self.single_frq)
        self.assertTrue(sig.ndim>1)


class TestAmpMod(unittest.TestCase):
    fps = 44100
    mod_idx = floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False, exclude_max=True)

    @composite
    @staticmethod
    def amp_mod_params(draw: DrawFn) -> tuple[float, float, float]:
        f_c = draw(frequencies(min_value=1, max_value=TestAmpMod.fps//2))
        f_m = draw(frequencies(min_value=0, max_value=f_c-1))
        m = draw(TestAmpMod.mod_idx)
        return (f_c, f_m, m)

    @given(amp_mod_params())
    def test_ampmod(self, params: tuple[float, float, float]) -> None:
        sig = tools.ampmod(*params)
        self.assertIsInstance(sig, np.ndarray)
        self.assertEqual(sig.dtype, np.float64)
        self.assertLessEqual(abs(sig).max(), 1)


class TestMelHzConverte(unittest.TestCase):
    @given(integers(min_value=0, max_value=1000000))
    def test(self, frq: int) -> None:
        res = tools.mel_to_hz(tools.hz_to_mel(frq))
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.isclose(res, frq))


def signal(n_frames: int, n_channels: int) -> FloatArray:
    """Signal of shape ``(n_frames, n_channels)`` with unique frame values."""
    return np.arange(n_frames*n_channels,
                     dtype=np.double).reshape(n_frames, n_channels)


def n_trimmed(fps: int, duration: int | None) -> int:
    """Number of frames ``trim_ms`` removes for ``duration`` milliseconds."""
    return 0 if duration is None else (fps*duration + 500) // 1000


@composite
def trim_specs(draw: DrawFn, which: str | None = None) -> TrimSpec:
    """Draw ``(sig, fps, pre, post)`` that leave at least one frame.

    Args:
        which:  Which boundaries to trim. One of ``'pre'``, ``'post'``, or
                ``'both'``. Drawn if not given.
    """
    fps = draw(integers(min_value=1, max_value=8000))
    which = which or draw(sampled_from(('pre', 'post', 'both')))
    durations = integers(min_value=1, max_value=200)
    pre = draw(durations) if which in ('pre', 'both') else None
    post = draw(durations) if which in ('post', 'both') else None
    n_frames = (n_trimmed(fps, pre) + n_trimmed(fps, post)
                + draw(integers(min_value=1, max_value=100)))
    return signal(n_frames, draw(integers(1, 3))), fps, pre, post


@composite
def overtrim_specs(draw: DrawFn) -> TrimSpec:
    """Draw ``(sig, fps, pre, post)`` that leave no frame at all."""
    fps = draw(integers(min_value=1, max_value=8000))
    n_frames = draw(integers(min_value=1, max_value=100))
    n_cut = n_frames + draw(integers(min_value=0, max_value=50))
    n_pre = draw(integers(min_value=0, max_value=n_cut))
    n_post = n_cut - n_pre
    pre = ceil(n_pre * 1000 / fps) if n_pre else None
    post = ceil(n_post * 1000 / fps) if n_post else None
    return signal(n_frames, draw(integers(1, 3))), fps, pre, post


class TestTrimMs(unittest.TestCase):
    @given(trim_specs())
    def test_removes_requested_number_of_frames(self, spec: TrimSpec) -> None:
        sig, fps, pre, post = spec
        out = tools.trim_ms(sig, fps, pre, post)
        self.assertEqual(out.shape[0], sig.shape[0] - n_trimmed(fps, pre)
                         - n_trimmed(fps, post))

    @given(trim_specs())
    def test_keeps_the_expected_frames(self, spec: TrimSpec) -> None:
        sig, fps, pre, post = spec
        start = n_trimmed(fps, pre)
        stop = sig.shape[0] - n_trimmed(fps, post)
        out = tools.trim_ms(sig, fps, pre, post)
        self.assertTrue(np.array_equal(out, sig[start:stop]))

    @given(trim_specs())
    def test_output_is_two_dimensional_view(self, spec: TrimSpec) -> None:
        sig, fps, pre, post = spec
        out = tools.trim_ms(sig, fps, pre, post)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], sig.shape[1])
        self.assertTrue(np.shares_memory(out, sig))

    @given(trim_specs())
    def test_trims_all_channels_alike(self, spec: TrimSpec) -> None:
        sig, fps, pre, post = spec
        out = tools.trim_ms(sig, fps, pre, post)
        for idx, channel in enumerate(sig.T):
            expected = tools.trim_ms(channel[:, None], fps, pre, post)
            self.assertTrue(np.array_equal(out[:, idx, None], expected))

    @given(trim_specs('post'))
    def test_omitted_pre_keeps_the_first_frame(self, spec: TrimSpec) -> None:
        sig, fps, _, post = spec
        out = tools.trim_ms(sig, fps, post=post)
        self.assertTrue(np.array_equal(out[0], sig[0]))

    @given(trim_specs('pre'))
    def test_omitted_post_keeps_the_last_frame(self, spec: TrimSpec) -> None:
        sig, fps, pre, _ = spec
        out = tools.trim_ms(sig, fps, pre=pre)
        self.assertTrue(np.array_equal(out[-1], sig[-1]))

    @given(trim_specs())
    def test_accepts_numpy_integers(self, spec: TrimSpec) -> None:
        sig, fps, pre, post = spec
        np_pre = None if pre is None else np.int64(pre)
        np_post = None if post is None else np.int32(post)
        self.assertTrue(np.array_equal(
            tools.trim_ms(sig, np.int64(fps), np_pre, np_post),
            tools.trim_ms(sig, fps, pre, post)))

    def test_rounds_to_nearest_frame(self) -> None:
        sig = signal(1000, 2)
        self.assertEqual(tools.trim_ms(sig, 44100, pre=1).shape[0], 1000-44)
        self.assertEqual(tools.trim_ms(sig, 1000, pre=100).shape[0], 900)

    def test_rounds_half_up(self) -> None:
        sig = signal(1000, 2)
        self.assertEqual(tools.trim_ms(sig, 5, pre=100).shape[0], 999)
        self.assertEqual(tools.trim_ms(sig, 5, pre=500).shape[0], 997)

    def test_sub_frame_duration_is_noop(self) -> None:
        sig = signal(1000, 2)
        self.assertTrue(np.array_equal(tools.trim_ms(sig, 100, 4, 4), sig))

    def test_keeps_single_remaining_frame(self) -> None:
        sig = signal(1000, 2)
        out = tools.trim_ms(sig, 1000, 400, 599)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(np.array_equal(out[0], sig[400]))

    @given(overtrim_specs())
    def test_raises_on_empty_result(self, spec: TrimSpec) -> None:
        sig, fps, pre, post = spec
        with self.assertRaises(ValueError):
            tools.trim_ms(sig, fps, pre, post)

    def test_raises_without_durations(self) -> None:
        with self.assertRaises(ValueError):
            tools.trim_ms(signal(1000, 2), 1000)

    @given(sampled_from((1, 3, 4)))
    def test_raises_on_wrong_dimensions(self, ndim: int) -> None:
        sig = np.zeros((4,)*ndim, dtype=np.double)
        with self.assertRaises(ValueError):
            tools.trim_ms(sig, 1000, pre=1)

    @given(sampled_from((0.5, 12.5, np.float64(10.0), '10', b'10')))
    def test_raises_on_non_integer_duration(self, duration: object) -> None:
        sig = signal(1000, 2)
        with self.assertRaises(TypeError):
            tools.trim_ms(sig, 1000, pre=duration)    # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            tools.trim_ms(sig, 1000, post=duration)   # type: ignore[arg-type]

    @given(integers(min_value=-1000, max_value=0))
    def test_raises_on_non_positive_duration(self, duration: int) -> None:
        sig = signal(1000, 2)
        with self.assertRaises(ValueError):
            tools.trim_ms(sig, 1000, pre=duration)
        with self.assertRaises(ValueError):
            tools.trim_ms(sig, 1000, post=duration)

    @given(integers(min_value=-1000, max_value=0))
    def test_raises_on_non_positive_fps(self, fps: int) -> None:
        sig = signal(1000, 2)
        with self.assertRaises(ValueError):
            tools.trim_ms(sig, fps, pre=100)

    @given(sampled_from((0.5, 44100.0, np.float64(1000.0), '1000')))
    def test_raises_on_non_integer_fps(self, fps: object) -> None:
        sig = signal(1000, 2)
        with self.assertRaises(TypeError):
            tools.trim_ms(sig, fps, pre=100)          # type: ignore[arg-type]


if __name__ == '__main__':
    unittest.main()
