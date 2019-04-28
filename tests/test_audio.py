#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
import numpy as _np

from apollon.audio import loadwav


class Test_ModulAudio(unittest.TestCase):
    def setUp(self):
        self.x = loadwav('/Users/michael/audio/beat.wav')

    def test_AudioDataAttributes(self):
        c = isinstance(self.x.fs, int)
        self.assertTrue(c)

        c = isinstance(self.x.data, _np.ndarray)
        self.assertTrue(c)

if __name__ == '__main__':
    unittest.main()
