#!/usr/bin/python3
# -*- coding: utf-8 -*-

from pathlib import Path
import unittest

from apollon.io import WaveFileAccessControl


class Foo:
    file = WaveFileAccessControl()
    def __init__(self, fname):
        self.file = fname


class Test_ModulIO(unittest.TestCase):
    def setUp(self):
        self.invalid_fname = 34

        self.not_existing_file = './xxx.py'
        self.not_a_file = '.'
        self.not_a_wav_file = '../../README.md'

    def test_InvalidFileNames(self):

        with self.assertRaises(TypeError):
            x = Foo(self.invalid_fname)

        with self.assertRaises(FileNotFoundError):
            x = Foo(self.not_existing_file)

        with self.assertRaises(IOError):
            x = Foo(self.not_a_file)
            x = Foo(self.not_a_wav_file)


if __name__ == '__main__':
    unittest.main()
