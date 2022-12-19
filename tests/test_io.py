#!/usr/bin/python3
"""apollon/tests/test_io.py
Test cases for IO module.
"""
import json
from pathlib import Path
import unittest

import numpy as np
from hypothesis import given
import hypothesis.extra.numpy as htn

import apollon.io as aio


class TestEncodeNdarray(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_encode(self, arr):
        encoded = aio.encode_ndarray(arr)
        self.assertTrue('__ndarray__' in encoded)
        self.assertTrue(encoded['__ndarray__'])
        self.assertTrue('__dtype__' in encoded)
        self.assertTrue(isinstance(encoded['__dtype__'], str))
        self.assertTrue('data' in encoded)
        self.assertTrue(isinstance(encoded['data'], list))


class TestDecodeNdarray(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_arrays(self, arr):
        restored = aio.decode_ndarray(aio.encode_ndarray(arr))
        self.assertTrue(arr.dtype.type is restored.dtype.type)
        self.assertTrue(arr.shape == restored.shape)
        self.assertTrue(np.allclose(arr, restored,
            rtol=0, atol=0, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
