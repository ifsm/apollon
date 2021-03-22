"""test_segment.py
"""
import unittest

from hypothesis import given
from hypothesis.strategies import integers, data, composite, SearchStrategy
from hypothesis.extra.numpy import array_shapes
import numpy as np

from apollon.audio import AudioFile
from apollon.segment import Segments, Segmentation, SegmentationParams


MAX_NSEGS = 345    # cannot pass instance attribute to method decorator


def _valid_nfr() -> SearchStrategy:
    return integers(min_value=2,
                    max_value=2000000)

def _valid_nps(n_frames: int) -> SearchStrategy:
    return integers(min_value=2,
                    max_value=n_frames)

def _valid_nol(n_perseg: int) -> SearchStrategy:
    return integers(min_value=1,
                    max_value=n_perseg-1)

@composite
def valid_nfr_nps_nol(draw) -> tuple:
    nfr = draw(_valid_nfr())
    nps = draw(_valid_nps(nfr))
    nol = draw(_valid_nol(nps))
    return nfr, nps, nol


class TestSegments(unittest.TestCase):
    def setUp(self) -> None:
        seg_params = SegmentationParams(1024, 512, extend=True, pad=True)
        self.segs = Segments(seg_params, np.empty((1024, 30)))

    @given(valid_nfr_nps_nol())
    def test_bounds_idx0_negative(self, shape) -> None:
        nfr, nps, nol = shape
        params = SegmentationParams(nps, nol, extend=True, pad=True)
        segs = Segments(params, np.empty((nps, nfr)))
        start, stop = segs.bounds(0)
        self.assertLess(start, 0)
        self.assertEqual(start, -(nps//2))    # parens because of floor div

    @given(integers(max_value=-1))
    def test_center_idx_gteq_zero(self, seg_idx) -> None:
        with self.assertRaises(IndexError):
            self.segs.bounds(seg_idx)

    @given(data())
    def test_center_idx_lt_nsegs(self, data) -> None:
        seg_idx = data.draw(integers(min_value=self.segs.n_segs))
        with self.assertRaises(IndexError):
            self.segs.bounds(seg_idx)

    @given(integers(max_value=-1))
    def test_bounds_idx_gteq_zero(self, seg_idx) -> None:
        with self.assertRaises(IndexError):
            bounds = self.segs.bounds(seg_idx)

    @given(data())
    def test_bounds_idx_lt_nsegs(self, data) -> None:
        seg_idx = data.draw(integers(min_value=self.segs.n_segs))
        with self.assertRaises(IndexError):
            bounds = self.segs.bounds(seg_idx)

    def test_data(self):
        seg_data = self.segs.data
        self.assertIsInstance(seg_data, np.ndarray)

    def test_extend_true(self) -> None:
        self.assertEqual(self.segs._offset, 0)

    def test_extend_false(self) -> None:
        n_perseg = 1024
        n_overlap = 512
        n_segs = 30
        seg_params = SegmentationParams(n_perseg, n_overlap, extend=False,
                                        pad=True)
        segs = Segments(seg_params, np.empty((n_perseg, n_segs)))
        self.assertEqual(segs._offset, n_perseg//2)


class TestSegmentation(unittest.TestCase):
    """
       nps -> n_perseg
       nol -> n_overlap
       gt  -> greater than
       lt  -> less than
       eq  -> eqaul to
    """
    def setUp(self) -> None:
        self.snd = AudioFile('audio/beat_5ch.wav')

    @given(integers(max_value=0))
    def test_nps_gt_zero(self, n_perseg) -> None:
        with self.assertRaises(ValueError):
            cutter = Segmentation(n_perseg, 1)

    @given(data())
    def test_nps_lteq_nframes(self, data) -> None:
        n_perseg = data.draw(integers(min_value=self.snd.n_frames+1))
        cutter = Segmentation(n_perseg, 1)
        with self.assertRaises(ValueError):
            cutter.transform(self.snd.data.squeeze())

    @given(data())
    def test_nol_gt_zero(self, data) -> None:
        nov_min_max = (None, 0)
        n_perseg = data.draw(self._valid_nps())
        n_overlap = data.draw(integers(*nov_min_max))
        with self.assertRaises(ValueError):
            cutter = Segmentation(n_perseg, n_overlap)

    @given(data())
    def test_nol_lt_nps(self, data) -> None:
        n_perseg = data.draw(self._valid_nps())
        n_overlap = data.draw(integers(min_value=n_perseg))
        with self.assertRaises(ValueError):
            cutter = Segmentation(n_perseg, n_overlap)

    @given(data())
    def test_inp_lt_three(self, data) -> None:
        n_perseg = data.draw(self._valid_nps())
        n_overlap = data.draw(_valid_nol(n_perseg))
        inp_shape = data.draw(array_shapes(min_dims=3))
        inp = np.empty(inp_shape)
        cutter = Segmentation(n_perseg, n_overlap)
        with self.assertRaises(ValueError):
            segs = cutter.transform(inp)

    @given(integers(min_value=2, max_value=1000))
    def test_inp2d_only_one_col(self, n_cols) -> None:
        n_frames = 1000
        n_perseg = 50
        n_overlap = 10
        inp = np.empty((n_frames, n_cols))
        cutter = Segmentation(n_perseg, n_overlap)
        with self.assertRaises(ValueError):
            segs = cutter.transform(inp)

    def _valid_nps(self):
        return _valid_nps(self.snd.n_frames)


if __name__ == '__main__':
    unittest.main()
