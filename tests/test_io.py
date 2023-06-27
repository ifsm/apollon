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

from apollon.io.io import WavFileAccessControl
from apollon.io.json import decode_ndarray, encode_ndarray

class MockFileLoader:
    file = WavFileAccessControl()
    def __init__(self, fname):
        self.file = fname


class TestWavFileAccessControl(unittest.TestCase):
    def setUp(self):
        self.invalid_fname = 34
        self.not_existing_file = './xxx.py'
        self.not_a_file = '.'
        self.not_a_wav_file = '../../README.md'

    def test_invalid_file_names(self):
        with self.assertRaises(TypeError):
            MockFileLoader(self.invalid_fname)

        with self.assertRaises(FileNotFoundError):
            MockFileLoader(self.not_existing_file)

        with self.assertRaises(IOError):
            MockFileLoader(self.not_a_file)
            MockFileLoader(self.not_a_wav_file)


class TestEncodeNdarray(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_encode(self, arr):
        encoded = encode_ndarray(arr)
        self.assertTrue('__ndarray__' in encoded)
        self.assertTrue(encoded['__ndarray__'])
        self.assertTrue('__dtype__' in encoded)
        self.assertTrue(isinstance(encoded['__dtype__'], str))
        self.assertTrue('data' in encoded)
        self.assertTrue(isinstance(encoded['data'], list))


class TestDecodeNdarray(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_arrays(self, arr):
        restored = decode_ndarray(encode_ndarray(arr))
        self.assertTrue(arr.dtype.type is restored.dtype.type)
        self.assertTrue(arr.shape == restored.shape)
        self.assertTrue(np.allclose(arr, restored,
            rtol=0, atol=0, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
