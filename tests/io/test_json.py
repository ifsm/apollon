import json
import unittest

from apollon.io.json import load_schema

class TestLoadSchema(unittest.TestCase):
    def setUp(self):
        pass

    def test_load(self):
        schema = load_schema('ndarray')
        self.assertTrue(isinstance(schema, dict))
