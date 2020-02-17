import unittest
import numpy as np

from apollon.audio import AudioFile


class TestAudioFileMono(unittest.TestCase):
    def setUp(self):
        self.snd_mono = AudioFile('audio/beat.wav', norm=False, mono=True)
        self.snd_multi = AudioFile('audio/beat_5ch.wav', norm=False, mono=True)
        self.comp_mono = self.snd_mono._file.read(always_2d=True)
        self.snd_mono._file.seek(0)

    def test_data_property(self):
        print(self.snd_multi.data.shape, self.comp_mono.shape)
        self.assertTrue(np.array_equal(self.snd_mono.data, self.comp_mono))
        self.assertTrue(np.array_equal(self.snd_multi.data, self.comp_mono))

    def tearDown(self):
        self.snd_mono.close()


class TestAudioFileMultiChannel(unittest.TestCase):
    def setUp(self):
        self.snd_mono = AudioFile('audio/beat.wav', mono=False)
        self.snd_multi = AudioFile('audio/beat_5ch.wav', mono=False)
        self.comp_mono = self.snd_mono._file.read(always_2d=True)
        self.comp_multi = self.snd_multi._file.read(always_2d=True)
        self.snd_mono._file.seek(0)
        self.snd_multi._file.seek(0)

    def test_data_property(self):
        self.assertTrue(np.array_equal(self.snd_mono.data, self.comp_mono))
        self.assertTrue(np.array_equal(self.snd_multi.data, self.comp_multi))

    def tearDown(self):
        self.snd_mono.close()
        self.snd_multi.close()


class TestAudioFileMonoNormalize(unittest.TestCase):
    def setUp(self):
        self.snd_mono = AudioFile('audio/beat.wav', mono=True, norm=True)
        self.snd_multi = AudioFile('audio/beat_5ch.wav', mono=True, norm=True)

    def test_data_property(self):
        self.assertTrue(np.less_equal(self.snd_mono.data, 1.0).all())
        self.assertTrue(np.greater_equal(self.snd_mono.data, -1.0).all())
        self.assertTrue(np.less_equal(self.snd_multi.data, 1.0).all())
        self.assertTrue(np.greater_equal(self.snd_multi.data, -1.0).all())

    def tearDown(self):
        self.snd_mono.close()
        self.snd_multi.close()


class TestAudioFileMultiChNormalize(unittest.TestCase):
    def setUp(self):
        self.snd_mono = AudioFile('audio/beat.wav', mono=False, norm=True)
        self.snd_multi = AudioFile('audio/beat_5ch.wav', mono=False, norm=True)

    def test_data_property(self):
        self.assertTrue(np.less_equal(self.snd_mono.data, 1.0).all())
        self.assertTrue(np.greater_equal(self.snd_mono.data, -1.0).all())
        self.assertTrue(np.less_equal(self.snd_multi.data, 1.0).all())
        self.assertTrue(np.greater_equal(self.snd_multi.data, -1.0).all())

    def tearDown(self):
        self.snd_mono.close()
        self.snd_multi.close()
"""
def test_fti16(self):
    res = fti16(self.snd_mono.data)
    self.assertTrue(isinstance(res, Array))
    self.assertTrue(res.dtype == 'int16')
    self.assertTrue(self.snd_mono.data.shape == res.shape)
"""

if __name__ == '__main__':
    unittest.main()
