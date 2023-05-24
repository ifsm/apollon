import unittest

from hypothesis import given
from hypothesis.strategies import floats, integers, tuples, SearchStrategy
from apollon import fractal

Params = tuple[float, float, float]
States = tuple[float, float, float]

def states() -> SearchStrategy:
    return tuples(floats(), floats(), floats())

def params() -> SearchStrategy:
    return tuples(floats(), floats(), floats())

class TestLorenzSystem(unittest.TestCase):
    @given(states(), params())
    def test_lorenz_system(self, state: States, params: Params
                           ) -> None:
        fractal.lorenz_system(state, *params)


class TestLorenzAttractor(unittest.TestCase):
    @given(integers(min_value=1, max_value=5000), params(), states(),
           floats(min_value=0.01, max_value=10))
    def test_lorenz_attractor(self, n_samples: int, params: Params,
                              init: States, diff: float) -> None:
        fractal.lorenz_attractor(n_samples, *params, init, diff)
