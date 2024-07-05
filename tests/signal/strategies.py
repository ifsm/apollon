from apollon.typing import FloatArray

from hypothesis import strategies as st
from hypothesis import given

import numpy as np


__all__ = [
    "frequencies",
    "samplerates",
    "fftsizes",
    "triang_const_frqs",
    "triangspecs",
    "filterspecs",
]


def frequencies(min_value: float = 0, max_value: float | None = None) -> st.SearchStrategy[float]:
    if min_value < 0:
        raise ValueError("Value of lower frequency bound less than 0")
    return st.floats(min_value=min_value, max_value=max_value, allow_infinity=False, allow_nan=False)

def samplerates() -> st.SearchStrategy[int]:
    return st.integers(min_value=1, max_value=96000)

def fftsizes(min_value: int = 1) -> st.SearchStrategy[int]:
    return st.integers(min_value=min_value, max_value=48000)


@st.composite
def triang_const_frqs(
    draw: st.DrawFn, fps: int, n_fft: int
) -> tuple[float, float, float]:
    if n_fft < 3:
        raise ValueError("n_fft less than 4")

    frqs = np.fft.rfftfreq(n_fft, 1 / fps)

    if n_fft == 3:
        return tuple(frqs)
    else:
        center = frqs.size // 2
        low = draw(st.sampled_from(frqs[:center]))  # type: ignore
        high = draw(st.sampled_from(frqs[center + 1 :]))  # type: ignore

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
    f_max = (n_fft + 1) // 2 if n_fft % 2 else n_fft // 2 + 1
    low = draw(frequencies(max_value=f_max // 4))
    high = draw(frequencies(min_value=f_max // 2, max_value=f_max))
    nflt = draw(st.integers(min_value=1, max_value=100))
    return (low, high, nflt, fps, n_fft)
