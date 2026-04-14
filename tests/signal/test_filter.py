from unittest import TestCase

from apollon.typing import FloatArray
from apollon.signal.filter import triang, triang_filter_bank
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
