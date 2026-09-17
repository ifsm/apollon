from unittest import TestCase

from apollon.typing import FloatArray
from apollon.signal.filter import preemphasis, triang, triang_filter_bank
from apollon.signal.tools import sinusoid
from hypothesis import strategies as st
from hypothesis import assume, given
import numpy as np

from .strategies import fftsizes, frequencies, samplerates


@st.composite
def triang_const_frqs(draw: st.DrawFn, fps: int, n_fft: int) -> tuple[float, float, float]:
    if n_fft < 3:
        raise ValueError("n_fft less than 4")

    frqs = np.fft.rfftfreq(n_fft, 1/fps)

    if n_fft == 3:
        return tuple(frqs)
    else:
        center = frqs.size // 2
        low = draw(st.sampled_from(frqs[:center]))      # type: ignore
        high = draw(st.sampled_from(frqs[center+1:]))   # type: ignore

    return low, center, high


@st.composite
def filterspecs(draw: st.DrawFn) -> tuple[int, int, FloatArray]:
    fps = draw(samplerates())
    n_fft = draw(fftsizes(min_value=4))
    items = draw(st.lists(triang_const_frqs(fps, n_fft), min_size=1, max_size=50))
    return (fps, n_fft, np.array(items))


@st.composite
def triangspec(draw: st.DrawFn) -> tuple[float, float, int, int, int]:
    fps = draw(samplerates())
    n_fft = draw(fftsizes(min_value=4))
    f_max = (n_fft+1)//2 if n_fft % 2 else n_fft//2+1
    low = draw(frequencies(max_value=f_max//4))
    high = draw(frequencies(min_value=f_max//2, max_value=f_max))
    nflt = draw(st.integers(min_value=1, max_value=100))
    return (low, high, nflt, fps, n_fft)

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


class TestTriang(TestCase):
    @given(
        filterspecs(),
        st.just((0,1,0))
    )
    def test_triang(self, filterspec: tuple[int, int, FloatArray], amps: tuple[float, float, float]) -> None:
        fps, size, frqs = filterspec
        res = triang(fps, size, frqs, amps)
        self.assertEqual(res.shape[0], frqs.shape[0])

    @given(st.integers(min_value=0, max_value=2))
    def test_bad_nfft(self, n_fft: int) -> None:
        with self.assertRaises(ValueError):
            triang(1, n_fft, np.array([[0, 1, 2]]))


class TestTriangFilterBank(TestCase):
    @given(triangspec())
    def test_triang_filter_bank(self, spec: tuple[float, float, int, int, int]) -> None:
        low, high, n_filters, fps, size = spec
        if low < high:
            if high > fps//2:
                with self.assertRaises(ValueError):
                    fb = triang_filter_bank(low, high, n_filters, fps, size)
            else:
                fb = triang_filter_bank(low, high, n_filters, fps, size)
                self.assertIsInstance(fb, np.ndarray)
        else:
            with self.assertRaises(ValueError):
                fb = triang_filter_bank(low, high, n_filters, fps, size)
                self.assertIsInstance(fb, np.ndarray)
