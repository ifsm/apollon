#!/usr/bin/env python3

import unittest
import numpy as np

from apollon.audio import load_audio, fti16
from apollon.types import Array


class Test_ModulAudio(unittest.TestCase):
    def setUp(self):
        self.snd = load_audio('audio/beat.wav')

    def test_AudioData(self):
        self.assertTrue(isinstance(self.snd.fps, int))
        self.assertTrue(isinstance(self.snd.data, Array))

    def test_fti16(self):
        res = fti16(self.snd.data)
        self.assertTrue(isinstance(res, Array))
        self.assertTrue(res.dtype == 'int16')
        self.assertTrue(self.snd.data.shape == res.shape)


if __name__ == '__main__':
    unittest.main()
