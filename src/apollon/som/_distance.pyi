from numbers import Real
from typing import Sequence
from apollon.typing import FloatArray


def hellinger(prob_a: Sequence[Real], prob_b: Sequence[Real]) -> float:
    ...

def hellinger_stm(sm_A: FloatArray, sm_B: FloatArray) -> float:
    ...
