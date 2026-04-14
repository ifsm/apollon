from hypothesis import strategies as st
from hypothesis import assume, given


__all__ = ["frequencies", "samplerates", "fftsizes"]


def frequencies(min_value: float = 0, max_value: float | None = None) -> st.SearchStrategy[float]:
    if min_value < 0:
        raise ValueError("Value of lower frequency bound less than 0")
    return st.floats(min_value=min_value, max_value=max_value, allow_infinity=False, allow_nan=False)

def samplerates() -> st.SearchStrategy[int]:
    return st.integers(min_value=1, max_value=96000)

def fftsizes(min_value: int = 1) -> st.SearchStrategy[int]:
    return st.integers(min_value=min_value, max_value=48000)
