import unittest
import numpy as np

from hypothesis import given, assume
from hypothesis import strategies as st
import hypothesis.extra.numpy as htn

from apollon.types import Array
from apollon.signal import features
from apollon.signal import tools
from apollon._defaults import SPL_REF

finite_float_arrays = htn.arrays(np.float,
        htn.array_shapes(min_dims=2, max_dims=2, min_side=2),
        elements = st.floats(allow_nan=False, allow_infinity=False))
"""
class TestCdim(unittest.TestCase):
    def setUp(self):
        self.data = sinusoid((300, 600), [.2, .1], fps=3000, noise=None)
        self.ecr = features.cdim(self.data, delay=14, m_dim=80, n_bins=1000,
                scaling_size=10, mode='bader')

    def test_cdim_returns_array(self):
        self.assertTrue(isinstance(self.ecr, Array))

    def test_cdim_gt_zero(self):
        self.assertTrue(np.all(self.ecr > 0))
"""

class TestEnergy(unittest.TestCase):
    @given(finite_float_arrays)
    def test_energy_positive(self, test_sig):
        res = features.energy(test_sig) >= 0
        self.assertTrue(res.all())


class TestSpl(unittest.TestCase):
    def setUp(self):
        self.lower_bound = np.array([SPL_REF, SPL_REF*0.1])
        self.range = np.array([SPL_REF+1e-6, 1.0])

    def test_spl_lower_bound(self):
        cnd = np.all(features.spl(self.lower_bound) == 0)
        self.assertTrue(cnd)

    def test_spl_range(self):
        cnd = np.all(features.spl(self.range) > 0)
        self.assertTrue(cnd)

if __name__ == '__main__':
    unittest.main()
