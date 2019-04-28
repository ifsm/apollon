#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""hmm_test.py

(c) Michael Blaß 2016

Unit test for HMM implementation."""


import unittest
import numpy as _np

from apollon.hmm.hmm_base import is_tpm


class TestHMM_utilities(unittest.TestCase):
    def setUp(self):
        # Arbitrary transition probability matrix
        self.A = _np.array([[1., 0, 0], [.2, .3, .5], [.1, .3, .6]])

        # Wrong number of dimensions
        self.B1 = _np.array([1., 0, 0, 0])
        self.B2 = _np.array([[[1., 0, 0], [.2, .3, .5], [.1, .3, .6]]])

        # Not quadratic
        self.C1 = _np.array([[1., 0, 0], [.2, .3, .5]])
        self.C2 = _np.array([[1.0], [.5, .5], [.2, .8]])

        # Rows do not sum up to one
        self.D = _np.array([[.2, .3, .5], [.5, .4, .2], [1., 0, 0]])

    def test_true_tpm(self):
        self.assertTrue(is_tpm(self.A), True)

if __name__ == '__main__':
    unittest.main()
