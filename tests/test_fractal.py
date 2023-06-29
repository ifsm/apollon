# pylint: disable = missing-function-docstring,missing-class-docstring
from functools import partial
import unittest

from hypothesis import given
from hypothesis.strategies import floats, integers, tuples, SearchStrategy
import numpy as np
from apollon import fractal

Params = tuple[float, float, float]
States = tuple[float, float, float]

rfloats = partial(floats, allow_nan=False, allow_infinity=False)
unit_floats = rfloats(min_value=0.0, max_value=1.0)

def states() -> SearchStrategy:
    return tuples(unit_floats, unit_floats, unit_floats)

def params() -> SearchStrategy:
    return tuples(rfloats(min_value=8.0, max_value=12.0),
                  rfloats(min_value=26.0, max_value=30.0),
                  rfloats(min_value=8/3-0.5, max_value=8/3+0.5))

class TestLorenzSystem(unittest.TestCase):
    @given(states(), params())
    def test_lorenz_system(self, state: States, params: Params
                           ) -> None:
        fractal.lorenz_system(state, *params)


class TestLorenzAttractor(unittest.TestCase):
    @given(integers(min_value=1, max_value=1000), params(), states())
    def test_lorenz_attractor(self, n_samples: int, params: Params,
                              init: States) -> None:
        res = fractal.lorenz_attractor(n_samples, *params, init, 0.01)
        self.assertFalse(np.isnan(res).any())
        self.assertFalse(np.isinf(res).any())
