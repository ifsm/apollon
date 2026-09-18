from typing import Literal
from unittest import TestCase

from apollon.typing import FloatArray
from apollon.signal.filter import lifter, mel_space, preemphasis, triangular_filter_bank
from apollon.signal.spectral import Stft
from apollon.signal.tools import sinusoid
from hypothesis import strategies as st
from hypothesis import assume, given
import numpy as np

from .strategies import fftsizes, samplerates


Scale = Literal["mel", "hz"]


@st.composite
def bankspec(draw: st.DrawFn) -> tuple[FloatArray, float, float, int, Scale]:
    """Draw a frequency axis together with cut-off frequencies taken from it.

    Drawing the cut-offs from the axis itself keeps them within its range
    whatever sample rate was drawn. ``n_filters`` is kept below half the
    number of bins in range, so that most specs resolve every filter.
    """
    fps = draw(samplerates())
    size = draw(fftsizes(min_value=4, max_value=1024))
    frqs = np.fft.rfftfreq(size, 1/fps)
    assume(frqs.size >= 4)
    low = draw(st.integers(min_value=0, max_value=frqs.size-2))
    high = draw(st.integers(min_value=low+1, max_value=frqs.size-1))
    n_filters = draw(st.integers(min_value=1, max_value=max(1, (high-low)//2)))
    scale = draw(st.sampled_from(("mel", "hz")))
    return frqs, float(frqs[low]), float(frqs[high]), n_filters, scale


def centers(low: float, high: float, n_filters: int, scale: Scale) -> FloatArray:
    """Return the centre frequencies of a filter bank in Hz."""
    if scale == "mel":
        return mel_space(low, high, n_filters+2).ravel()[1:-1]
    return np.linspace(low, high, n_filters+2)[1:-1]


def signals(min_size: int = 2, max_size: int = 200) -> st.SearchStrategy[FloatArray]:
    elements = st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False,
                         allow_nan=False)
    return st.lists(elements, min_size=min_size, max_size=max_size).map(
            lambda items: np.array(items, dtype=np.double))


class TestPreemphasis(TestCase):
    @given(signals(), st.floats(min_value=0.0, max_value=1.0))
    def test_difference_equation(self, inp: FloatArray, coef: float) -> None:
        out, _ = preemphasis(inp, coef)
        self.assertTrue(np.allclose(out[1:], inp[1:] - coef*inp[:-1]))

    @given(signals(), st.floats(min_value=0.0, max_value=1.0))
    def test_extrapolated_first_sample(self, inp: FloatArray, coef: float) -> None:
        out, _ = preemphasis(inp, coef)
        self.assertAlmostEqual(out[0], inp[0] - coef*(2*inp[0]-inp[1]))

    def test_accepts_sinusoid(self) -> None:
        inp = sinusoid(300, fps=1000)
        out, state = preemphasis(inp)
        self.assertEqual(out.shape, inp.shape)
        self.assertEqual(state.shape, (1,))

    @given(signals(min_size=3), st.floats(min_value=0.0, max_value=1.0),
           st.integers(min_value=2))
    def test_blockwise_equals_whole(self, inp: FloatArray, coef: float,
                                    split: int) -> None:
        assume(split < inp.size)
        whole, _ = preemphasis(inp, coef)
        head, state = preemphasis(inp[:split], coef)
        tail, _ = preemphasis(inp[split:], coef, prev=state)
        self.assertTrue(np.allclose(np.concatenate([head, tail]), whole))

    @given(signals(min_size=2, max_size=32), st.floats(min_value=0.0, max_value=1.0),
           st.integers(min_value=2, max_value=4))
    def test_filters_along_first_axis(self, inp: FloatArray, coef: float,
                                      n_channels: int) -> None:
        multi = inp[:, None] * np.arange(1, n_channels+1)
        out, state = preemphasis(multi, coef)
        self.assertEqual(out.shape, multi.shape)
        self.assertEqual(state.shape, (n_channels,))
        for idx, channel in enumerate(multi.T):
            expected, _ = preemphasis(channel, coef)
            self.assertTrue(np.allclose(out[:, idx], expected))

    def test_state_continues_the_signal(self) -> None:
        inp = np.arange(10, dtype=np.double)
        _, state = preemphasis(inp)
        self.assertEqual(state, inp[-1])

    def test_single_sample_without_prev(self) -> None:
        with self.assertRaises(ValueError):
            preemphasis(np.array([1.0]))

    def test_single_sample_with_prev(self) -> None:
        out, _ = preemphasis(np.array([1.0]), 0.5, prev=2.0)
        self.assertAlmostEqual(out[0], 0.0)

    def test_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            preemphasis(np.array([]), prev=0.0)

    @given(signals())
    def test_output_is_float64(self, inp: FloatArray) -> None:
        out, _ = preemphasis(inp.astype(np.float32))
        self.assertEqual(out.dtype, np.double)


class TestLifter(TestCase):
    @given(signals(), st.floats(min_value=1.0, max_value=100.0))
    def test_matches_reference(self, inp: FloatArray, lift: float) -> None:
        idx = np.arange(1, inp.shape[0]+1)
        expected = inp * (1 + lift/2 * np.sin(np.pi*idx/lift))
        self.assertTrue(np.allclose(lifter(inp, lift), expected))

    @given(signals())
    def test_zero_lift_is_identity(self, inp: FloatArray) -> None:
        out = lifter(inp, 0.0)
        self.assertTrue(np.array_equal(out, inp))
        self.assertIsNot(out, inp)

    @given(signals(), st.floats(min_value=1.0, max_value=100.0))
    def test_negative_lift_raises(self, inp: FloatArray, lift: float) -> None:
        with self.assertRaises(ValueError):
            lifter(inp, -lift)

    @given(signals())
    def test_output_is_float64(self, inp: FloatArray) -> None:
        self.assertEqual(lifter(inp.astype(np.float32), 22.0).dtype, np.double)
        self.assertEqual(lifter(inp.astype(int), 22.0).dtype, np.double)

    @given(signals())
    def test_accepts_1d(self, inp: FloatArray) -> None:
        self.assertEqual(lifter(inp, 22.0).shape, inp.shape)

    @given(signals(max_size=32), st.integers(min_value=2, max_value=4))
    def test_lifters_along_first_axis(self, inp: FloatArray, n_frames: int) -> None:
        frames = inp[:, None] * np.arange(1, n_frames+1)
        out = lifter(frames, 22.0)
        self.assertEqual(out.shape, frames.shape)
        for idx, frame in enumerate(frames.T):
            self.assertTrue(np.allclose(out[:, idx], lifter(frame, 22.0)))


class TestTriangularFilterBank(TestCase):

    fps = 16000
    size = 512

    @property
    def frqs(self) -> FloatArray:
        return np.fft.rfftfreq(self.size, 1/self.fps)

    @given(bankspec())
    def test_no_silently_dead_filter(
            self, spec: tuple[FloatArray, float, float, int, Scale]) -> None:
        """A filter that acts on no frequency at all must be reported rather
        than returned as a zero row. Zero rows used to arise from rounding the
        band edges to FFT bins, where two coinciding edges left ``np.interp``
        with a non-increasing ``xp``."""
        frqs, low, high, n_filters, scale = spec
        try:
            fbank = triangular_filter_bank(frqs, low, high, n_filters, scale)
        except ValueError:
            return
        self.assertTrue(np.all(fbank.any(axis=1)))

    @given(bankspec())
    def test_shape_and_range(
            self, spec: tuple[FloatArray, float, float, int, Scale]) -> None:
        frqs, low, high, n_filters, scale = spec
        try:
            fbank = triangular_filter_bank(frqs, low, high, n_filters, scale)
        except ValueError:
            assume(False)
            return
        self.assertEqual(fbank.shape, (n_filters, frqs.size))
        self.assertTrue(np.all(fbank >= 0.0))
        self.assertTrue(np.all(fbank <= 1.0))
        self.assertEqual(fbank.dtype, np.double)

    @given(st.integers(min_value=4, max_value=64))
    def test_no_indexing_past_the_last_bin(self, size: int) -> None:
        """Cut-off at the highest frequency of the axis. Rounding the edge to a
        bin index used to overshoot the last bin for odd ``size``."""
        frqs = np.fft.rfftfreq(size, 1/22050)
        fbank = triangular_filter_bank(frqs, 1.0, float(frqs.max()), 1)
        self.assertEqual(fbank.shape, (1, frqs.size))

    def test_partition_of_unity(self) -> None:
        """Neighbouring filters overlap by half, so their responses sum to one
        between the first and the last centre frequency. This follows from the
        shape of the filters and hence holds in either scale."""
        n_filters = 26
        low, high = 80.0, 7600.0
        for scale in ("mel", "hz"):
            with self.subTest(scale=scale):
                fbank = triangular_filter_bank(self.frqs, low, high, n_filters, scale)
                ctr = centers(low, high, n_filters, scale)
                inner = (self.frqs >= ctr[0]) & (self.frqs <= ctr[-1])
                self.assertTrue(np.allclose(fbank[:, inner].sum(axis=0), 1.0))

    def test_peak_is_bounded_but_not_exact(self) -> None:
        """Each apex sits on its exact centre frequency, which generally falls
        between two bins. The largest sampled weight is hence at most one, and
        need not reach it."""
        fbank = triangular_filter_bank(self.frqs, 80.0, 7600.0, 80)
        self.assertTrue(np.all(fbank.max(axis=1) <= 1.0))
        self.assertTrue(np.all(fbank.max(axis=1) > 0.0))

    def test_centers_ascend(self) -> None:
        fbank = triangular_filter_bank(self.frqs, 80.0, 7600.0, 26)
        self.assertTrue(np.all(np.diff(fbank.argmax(axis=1)) >= 0))

    def test_accepts_column_vector_axis(self) -> None:
        """``Stft.frqs`` is an Nx1 axis."""
        sxx = Stft(fps=self.fps, n_perseg=self.size,
                   n_overlap=self.size//2).transform(sinusoid(440, fps=self.fps))
        self.assertEqual(sxx.frqs.ndim, 2)
        from_column = triangular_filter_bank(sxx.frqs, 80.0, 7600.0, 26)
        self.assertTrue(np.array_equal(from_column,
                                       triangular_filter_bank(self.frqs, 80.0, 7600.0, 26)))

    def test_applies_to_a_spectrogram(self) -> None:
        """The bank maps a power spectrogram to one row per filter, and a
        sinusoid lands in the band holding its frequency."""
        sxx = Stft(fps=self.fps, n_perseg=self.size,
                   n_overlap=self.size//2).transform(sinusoid(440, fps=self.fps))
        fbank = triangular_filter_bank(sxx.frqs, 80.0, 7600.0, 26)
        mel = fbank @ sxx.power
        self.assertEqual(mel.shape, (26, sxx.n_segments))
        centers = mel_space(80.0, 7600.0, 28).ravel()[1:-1]
        self.assertAlmostEqual(centers[mel.mean(axis=1).argmax()], 440.0, delta=60.0)

    def test_two_dimensional_axis_raises(self) -> None:
        with self.assertRaises(ValueError):
            triangular_filter_bank(np.zeros((16, 3)), 80.0, 7600.0, 26)

    def test_empty_axis_raises(self) -> None:
        with self.assertRaises(ValueError):
            triangular_filter_bank(np.array([]), 80.0, 7600.0, 26)

    def test_negative_low_raises(self) -> None:
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, -1.0, 7600.0, 26)

    def test_low_not_below_high_raises(self) -> None:
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, 7600.0, 80.0, 26)
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, 80.0, 80.0, 26)

    def test_high_above_axis_raises(self) -> None:
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, 80.0, float(self.frqs.max())+1.0, 26)

    @given(st.integers(min_value=-10, max_value=0))
    def test_non_positive_n_filters_raises(self, n_filters: int) -> None:
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, 80.0, 7600.0, n_filters)

    def test_unknown_scale_raises(self) -> None:
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, 80.0, 7600.0, 26, scale="bark")  # type: ignore[arg-type]

    def test_hz_filters_peak_at_equidistant_centers(self) -> None:
        """The defining property of the "hz" scale: each filter peaks at the
        bin closest to its linearly spaced centre frequency. Mel spacing, in
        contrast, compresses the low frequencies."""
        fbank = triangular_filter_bank(self.frqs, 80.0, 7600.0, 26, scale="hz")
        expected = np.linspace(80.0, 7600.0, 28)[1:-1]
        nearest = np.abs(self.frqs[None, :]-expected[:, None]).argmin(axis=1)
        self.assertTrue(np.array_equal(fbank.argmax(axis=1), nearest))

        spacing = np.diff(expected)
        self.assertTrue(np.allclose(spacing, spacing[0]))
        mel_spacing = np.diff(centers(80.0, 7600.0, 26, "mel"))
        self.assertFalse(np.allclose(mel_spacing, mel_spacing[0]))

    def test_hz_bandwidth_is_constant(self) -> None:
        """Equidistant centres on an equidistant frequency axis put the same
        number of bins under every filter."""
        fbank = triangular_filter_bank(self.frqs, 80.0, 7600.0, 26, scale="hz")
        bins_per_filter = (fbank > 0).sum(axis=1)
        self.assertLessEqual(bins_per_filter.max()-bins_per_filter.min(), 1)

    def test_scales_differ(self) -> None:
        """Guard against the "hz" branch falling through to Mel spacing."""
        args = (self.frqs, 80.0, 7600.0, 26)
        self.assertFalse(np.allclose(triangular_filter_bank(*args, scale="hz"),
                                     triangular_filter_bank(*args, scale="mel")))

    def test_hz_unresolvable_filter_count_raises(self) -> None:
        """The dead-filter guard applies in either scale."""
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, 100.0, 140.0, 5, scale="hz")

    def test_unresolvable_filter_count_raises(self) -> None:
        """More filters than the axis can resolve is reported, not silently
        returned as zero rows."""
        with self.assertRaises(ValueError):
            triangular_filter_bank(self.frqs, 80.0, 7600.0, 200)
