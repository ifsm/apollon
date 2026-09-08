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
from hypothesis import strategies as st
import hypothesis.extra.numpy as htn

import apollon.io._json as jsonio
import apollon.io._numpy as numpyio
import apollon.io._pickle as pickleio


picklable = st.recursive(
    st.none() | st.booleans() | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: (st.lists(children) | st.tuples(children)
                       | st.dictionaries(st.text(), children)),
    max_leaves=10,
)


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


class TestDumpLoadNumpy(unittest.TestCase):
    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_roundtrip(self, arr):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'arr.npy'
            numpyio.dump_numpy(arr, path)
            restored = numpyio.load_numpy(path)
        self.assertEqual(arr.dtype, restored.dtype)
        self.assertEqual(arr.shape, restored.shape)
        self.assertTrue(np.array_equal(arr, restored, equal_nan=True))

    @given(htn.arrays(htn.complex_number_dtypes(), htn.array_shapes()))
    def test_roundtrip_complex(self, arr):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'arr.npy'
            numpyio.dump_numpy(arr, path)
            restored = numpyio.load_numpy(path)
        self.assertEqual(arr.dtype, restored.dtype)
        self.assertEqual(arr.shape, restored.shape)
        self.assertTrue(np.array_equal(arr, restored, equal_nan=True))


class TestDumpLoadPickle(unittest.TestCase):
    @given(picklable)
    def test_roundtrip(self, obj):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'obj.pkl'
            pickleio.dump_pickle(obj, path)
            restored = pickleio.load_pickle(path)
        self.assertEqual(obj, restored)

    @given(htn.arrays(htn.floating_dtypes(), htn.array_shapes()))
    def test_roundtrip_array(self, arr):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'arr.pkl'
            pickleio.dump_pickle(arr, path)
            restored = pickleio.load_pickle(path)
        # dtype.kind/itemsize rather than dtype equality: pickling a
        # non-native-byteorder array normalizes its byteorder tag on
        # unpickling even though the values are preserved correctly.
        self.assertEqual(arr.dtype.kind, restored.dtype.kind)
        self.assertEqual(arr.dtype.itemsize, restored.dtype.itemsize)
        self.assertTrue(np.array_equal(arr, restored, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
