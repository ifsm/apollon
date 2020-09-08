import unittest
import numpy as np

from apollon.audio import AudioFile

class TestAudioFile(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat.wav')

    def test_path_is_sting(self):
        self.assertIsInstance(self.snd.file_name, str)

    def test_hash(self):
        snd2 = AudioFile('audio/beat.wav')
        self.assertEqual(self.snd.hash, snd2.hash)


class TestAudioFileReadMono(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat.wav')
        self.ref = self.snd._file.read(always_2d=True)

    def test_read_raw_multi(self):
        data = self.snd.read(-1, norm=False, mono=False)
        self.assertTrue(np.array_equal(self.ref, data))

    def test_read_raw_mono(self):
        ref = self.ref.sum(axis=1, keepdims=True) / self.ref.shape[1]
        data = self.snd.read(norm=False, mono=True)
        self.assertTrue(np.array_equal(ref, data))

    def test_read_norm_multi(self):
        ref = self.ref / self.ref.max(axis=0, keepdims=True)
        data = self.snd.read(norm=True, mono=False)
        self.assertTrue(np.array_equal(ref, data))

    def test_read_norm_mono(self):
        ref = self.ref.sum(axis=1, keepdims=True) / self.ref.shape[1]
        ref /= self.ref.max()
        data = self.snd.read(norm=True, mono=True)
        self.assertTrue(np.array_equal(ref, data))

    def tearDown(self):
        self.snd.close()


class TestAudioFileReadMultiChannel(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat_5ch.wav')
        self.ref = self.snd._file.read(always_2d=True)

    def test_read_raw_multi(self):
        data = self.snd.read(norm=False, mono=False)
        self.assertTrue(np.array_equal(self.ref, data))

    def test_read_raw_mono(self):
        ref = self.ref.sum(axis=1, keepdims=True) / self.ref.shape[1]
        data = self.snd.read(norm=False, mono=True)
        self.assertTrue(np.array_equal(ref, data))

    def test_read_norm_multi(self):
        ref = self.ref / self.ref.max(axis=0, keepdims=True)
        data = self.snd.read(norm=True, mono=False)
        self.assertTrue(np.array_equal(ref, data))

    def test_read_norm_mono(self):
        ref = self.ref.sum(axis=1, keepdims=True) / self.ref.shape[1]
        ref /= self.ref.max()
        data = self.snd.read(norm=True, mono=True)
        self.assertTrue(np.array_equal(ref, data))

    def tearDown(self):
        self.snd.close()


"""
def test_fti16(self):
    res = fti16(self.snd_mono.data)
    self.assertTrue(isinstance(res, Array))
    self.assertTrue(res.dtype == 'int16')
    self.assertTrue(self.snd_mono.data.shape == res.shape)
"""

if __name__ == '__main__':
    unittest.main()
