#!/usr/bin/env python3

import unittest
import numpy as np

from apollon._defaults import SPL_REF
from apollon.signal import features
from apollon.signal import tools


class TestAmp(unittest.TestCase):
    def test_amp_at_1Pa(self):
        sig = np.array([[1.0]], dtype=np.float64)
        res = tools.amp(features.spl(sig))
        self.assertTrue(np.allclose(res, sig))
        self.assertEqual(res.dtype.name, "float64")

    def test_amp_at_threshold(self):
        sig = np.array([[SPL_REF]], dtype=np.float64)
        res = tools.amp(features.spl(sig))
        self.assertTrue(np.allclose(res, sig))
        self.assertEqual(res.dtype.name, "float64")


class TestSinusoid(unittest.TestCase):
    def setUp(self):
        self.single_frq = 100
        self.multi_frq = (100, 200, 300)
        self.single_amp = .3
        self.multi_amp = (0.5, .3, .2)

    def test_returns_2darray_on_scalar_frq(self):
        sig = tools.sinusoid(self.single_frq)
        self.assertTrue(sig.ndim>1)



if __name__ == '__main__':
    unittest.main()
