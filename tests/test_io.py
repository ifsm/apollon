#!/usr/bin/python3
"""apollon/tests/test_io.py
Test cases for IO module.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from hypothesis import given
import hypothesis.extra.numpy as htn

import apollon.io._json as jsonio


class TestEncodeNdarray(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_encode(self, arr):
        encoded = jsonio.encode_ndarray(arr)
        self.assertTrue('__ndarray__' in encoded)
        self.assertTrue(encoded['__ndarray__'])
        self.assertTrue('__dtype__' in encoded)
        self.assertTrue(isinstance(encoded['__dtype__'], str))
        self.assertTrue('data' in encoded)
        self.assertTrue(isinstance(encoded['data'], list))

    @given(htn.arrays(htn.complex_number_dtypes(), htn.array_shapes()))
    def test_encode_complex(self, arr):
        encoded = jsonio.encode_ndarray(arr)
        self.assertTrue('__ndarray__' in encoded)
        self.assertTrue(encoded['__ndarray__'])
        self.assertTrue('__dtype__' in encoded)
        self.assertTrue(isinstance(encoded['__dtype__'], str))
        self.assertTrue('data' in encoded)
        self.assertTrue(isinstance(encoded['data'], dict))
        self.assertTrue('real' in encoded['data'])
        self.assertTrue('imag' in encoded['data'])


class TestDecodeNdarray(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_arrays(self, arr):
        restored = jsonio.decode_ndarray(jsonio.encode_ndarray(arr))
        self.assertTrue(arr.dtype.type is restored.dtype.type)
        self.assertTrue(arr.shape == restored.shape)
        self.assertTrue(np.allclose(arr, restored,
            rtol=0, atol=0, equal_nan=True))

    @given(htn.arrays(htn.complex_number_dtypes(), htn.array_shapes()))
    def test_arrays_complex(self, arr):
        restored = jsonio.decode_ndarray(jsonio.encode_ndarray(arr))
        self.assertTrue(arr.dtype.type is restored.dtype.type)
        self.assertTrue(arr.shape == restored.shape)
        self.assertTrue(np.allclose(arr, restored,
            rtol=0, atol=0, equal_nan=True))

    def test_invalid_instance_raises_type_error(self):
        with self.assertRaises(TypeError):
            jsonio.decode_ndarray({})


class TestDumpLoadJson(unittest.TestCase):
    def test_roundtrip_complex_array(self):
        arr = np.array([1 + 2j, 3 - 4j, 0j])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'arr.json'
            jsonio.dump_json(arr, path)
            restored = jsonio.load_json(path)
        self.assertEqual(arr.dtype, restored.dtype)
        self.assertTrue(np.array_equal(arr, restored))


if __name__ == '__main__':
    unittest.main()
