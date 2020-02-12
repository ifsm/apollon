#!/usr/bin/python3
"""apollon/tests/test_io.py
Test cases for IO module.
"""
import json
from pathlib import Path
import unittest
from hypothesis import given
import hypothesis.extra.numpy as htn

import apollon.io as aio


class MockFileLoader:
    file = aio.WavFileAccessControl()
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

"""
class Test(unittest.TestCase):
    def test_encode_arrays(self, arr):
        trans = aio.dump_json(arr)
        restored = json.loads(trans, object_hook=aio.decode_array)
        data = np.allclose(trans, restored, atol=0, btol=0, equal_nan=True)
        self.assertTrue(data)
        self.assertTrue(arr.dtype.type is restored.dtype.type)
        self.assertTrue(arr.shape == restored.shape)
"""

class TestEncodeDecodeArray(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_arrays(self, arr):
        restored = aio.decode_array(aio.encode_array(arr))
        self.assertTrue(np.array_equal(arr, restored))


class TestDecode_array(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
