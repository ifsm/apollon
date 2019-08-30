#!/usr/bin/python3

import unittest
import numpy as np
import scipy as sp

from apollon.som import utilities


class TestSelfOrganizingMap(unittest.TestCase):
    def setUp(self):
        self.weights = np.load('data/test_weights.npy')
        self.inp = np.load('data/test_inp.npy')

    def test_best_match_computation(self):
        test_bmu = np.load('data/bmu_idx_euc.npy')
        test_err = np.load('data/bmu_err_euc.npy')
        bmu, err = utilities.best_match(self.weights, self.inp, 'euclidean')
        self.assertTrue(np.array_equiv(test_bmu, bmu))
        self.assertTrue(np.array_equiv(test_err, err))


if __name__ == '__main__':
    unittest.main()
