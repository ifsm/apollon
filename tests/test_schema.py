import unittest

import jsonschema
import numpy as np

from apollon import io

check_schema = jsonschema.Draft7Validator.check_schema

def validate(instance, schema):
    return jsonschema.validate(instance, schema,
                               jsonschema.Draft7Validator)



class TestCorrGramParams(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = io.json.load_schema('corrgram')
        self.corrgram = {'wlen': 100, 'n_delay': 20, 'total': True}

    def test_schema_is_valid(self):
        check_schema(self.schema)

    def test_fails_on_additional_property(self) -> None:
        self.corrgram['beer'] = None
        with self.assertRaises(jsonschema.ValidationError):
            validate(self.corrgram, self.schema)

    def test_fails_on_total_is_not_bool(self):
        self.corrgram['total'] = 'this_causes_an_error'
        with self.assertRaises(jsonschema.ValidationError):
            validate(self.corrgram, self.schema)


class TestCdimParams(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = io.json.load_schema('corrdim')
        self.corrdim = {'delay': 14, 'm_dim': 80, 'n_bins': 1000,
                     'scaling_size': 10}

    def test_schema_is_valid(self):
        check_schema(self.schema)

    def test_fails_on_additional_property(self) -> None:
        self.corrdim['beer'] = None
        with self.assertRaises(jsonschema.ValidationError):
            validate(self.corrdim, self.schema)


class TestDftParams(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = io.json.load_schema('dft_params')
        self.dft_params = {'fps': 44100, 'window': 'hamming', 'n_fft': None}

    def test_schema_is_valid(self):
        check_schema(self.schema)

    def test_fails_on_additional_property(self) -> None:
        self.dft_params['beer'] = None
        with self.assertRaises(jsonschema.ValidationError):
            validate(self.dft_params, self.schema)


class TestStftParams(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = io.json.load_schema('stft_params')
        self.stft_params = {'fps': 44100, 'window': 'hamming', 'n_fft': None,
                            'n_perseg': 1024, 'n_overlap': 512, 'extend': True,
                            'pad': True}

    def test_schema_is_valid(self) -> None:
        check_schema(self.schema)

    def test_fails_on_additional_property(self) -> None:
        self.stft_params['beer'] = None
        with self.assertRaises(jsonschema.ValidationError):
            validate(self.stft_params, self.schema)


class TestNdarray(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = io.json.load_schema('ndarray')
        self.data = np.arange(10.0)
        self.valid_array = io.json.encode_ndarray(self.data)

    def test_valid_schema(self) -> None:
        check_schema(self.schema)

    def test_pass_on_array(self) -> None:
        validate(self.valid_array, self.schema)

    def test_fails_on_addition_property(self) -> None:
        self.valid_array['a'] = 12
        with self.assertRaises(jsonschema.ValidationError):
            validate(self.valid_array, self.schema)

    def test_fails_on_data_is_not_array(self) -> None:
        self.valid_array['data'] = 12
        with self.assertRaises(jsonschema.ValidationError):
            validate(self.valid_array, self.schema)
