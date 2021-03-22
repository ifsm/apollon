import os
import string
import tempfile
import unittest

from hypothesis import given
from hypothesis.strategies import dictionaries, sampled_from, text
import jsonschema

from apollon.container import Params
from apollon import io


class TestParams(unittest.TestCase):
    def test_dict(self):
        params = Params()
        self.assertTrue(isinstance(params.to_dict(), dict))


class TestDumpJSON(unittest.TestCase):
    keys = text(sampled_from(string.ascii_letters), min_size=1, max_size=10)
    vals = text(sampled_from(string.ascii_letters), min_size=1, max_size=10)
    str_dicts = dictionaries(keys, vals)

    @given(str_dicts)
    def test_dump(self, inp) -> None:
        handle, path = tempfile.mkstemp(suffix='.json', text=True)
        io.json.dump(inp, path)
        res = io.json.load(path)
        out = [inp[k] == res[k] for k, v in res.items()]
        os.unlink(path)
        os.close(handle)
