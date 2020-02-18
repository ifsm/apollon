"""test_segment.py
"""
import unittest
from hypothesis import given
from hypothesis.strategies import integers

from apollon.audio import AudioFile
from apollon.segment import Segments


MAX_NSEGS = 345    # cannot pass instance attribute to method decorator

class TestComputeBoundsExpand(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat_5ch.wav')
        self.segs = Segments(self.snd, 2048, 1024)

    @given(integers(max_value=-1))
    def test_no_negative_indices(self, seg_idx):
        with self.assertRaises(IndexError):
            bounds = self.segs.compute_bounds(seg_idx)

    @given(integers(min_value=MAX_NSEGS+1))
    def test_index_exceeds_n_segs(self, seg_idx):
        with self.assertRaises(IndexError):
            bounds = self.segs.compute_bounds(seg_idx)

    def test_expand_starts_negative(self):
        start, stop = self.segs.compute_bounds(0)
        self.assertLess(start, 0)
        self.assertEqual(start, -self.segs.n_perseg//2)

    @given(integers(min_value=0, max_value=MAX_NSEGS))
    def test_returns_int(self, seg_idx):
        start, stop = self.segs.compute_bounds(seg_idx)
        self.assertTrue(isinstance(start, int))
        self.assertTrue(isinstance(stop, int))


class TestComputeDoundsNoExpands(unittest.TestCase):
    def setUp(self):
        self.snd = AudioFile('audio/beat_5ch.wav')
        self.segs = Segments(self.snd, 2048, 1024, expand=False)

    def test_expand_starts_at_zero(self):
        start, stop = self.segs.compute_bounds(0)
        self.assertEqual(start, 0)



if __name__ == '__main__':
    unittest.main()
