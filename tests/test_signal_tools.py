#!/usr/bin/env python3

import unittest
import numpy as np

from apollon._defaults import SPL_REF
from apollon.signal import features as feat
from apollon.signal import tools as st


class TestAmp(unittest.TestCase):
    def setUp(self):
        self.lower_bound = np.array([SPL_REF, SPL_REF*0.1])
        self.range = np.array([SPL_REF+1e-6, 1.0])

    def test_amp_lower_bound(self):
        res = st.amp(feat.spl(self.lower_bound))
        cnd = np.array_equal(res, np.array([SPL_REF, SPL_REF]))
        self.assertTrue(cnd)

    def test_amp_range(self):
        res = st.amp(feat.spl(self.range))
        cnd = np.allclose(res, self.range)
        self.assertTrue(cnd)


if __name__ == '__main__':
    unittest.main()
