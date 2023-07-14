"""
Time series segmentation utilities
"""

from typing import Generator

import numpy as np
from numpy.lib.stride_tricks import as_strided

from apollon.audio import AudioFile
from apollon.types import FloatArray, NDArray
from apollon.segment.models import SegmentationParams, Segment


class Segments:
    """Segement"""
    def __init__(self, params: SegmentationParams, segs: FloatArray) -> None:
        self._segs = segs
        self._params = params
        if self._params.extend:
            self._offset = 0
        else:
            self._offset = self._params.n_perseg // 2

    @property
    def data(self) -> FloatArray:
        """Return the raw segment data array"""
        return self._segs

    @property
    def n_segs(self) -> int:
        """Return number of segments"""
        return self._segs.shape[1]

    @property
    def n_perseg(self) -> int:
        """Number of samples per segment (window size)"""
        return self._params.n_perseg

    @property
    def n_overlap(self) -> int:
        """Number of overlapping samples in consecutive windows"""
        return self._params.n_overlap

    @property
    def step(self) -> int:
        """Distance between consecutive windows (hop size) in samples"""
        return self._params.n_perseg - self._params.n_overlap

    @property
    def params(self) -> SegmentationParams:
        """Return initialized parameters"""
        return self._params

    def center(self, seg_idx: int) -> int:
        """Return the center of segment ``seg_idx`` as frame number
        of the original signal.

        Args:
            seg_indx:  Segment index

        Returns:
            Center frame index
        """
        if not 0 <= seg_idx < self.n_segs:
            raise IndexError('Requested index out of range.')
        return seg_idx * self.step + self._offset

    def bounds(self, seg_idx: int) -> tuple[int, int]:
        """Return the frame numbers of the lower and upper bound
        of segment ``seg_idx``. Lower bound index is inclusive,
        upper bound index is exclusive.

        Args:
            seg_idx:  Segment index

        Returns:
            Lower and upper bound frame index
        """
        if not 0 <= seg_idx < self.n_segs:
            raise IndexError('Requested index out of range.')
        lob = self.center(seg_idx) - self._params.n_perseg // 2
        upb = lob + self._params.n_perseg
        return lob, upb

    def get(self, seg_idx: int) -> Segment:
        """Retrun segment ``seg_idx`` wrapped in an ``Segment`` object.

        Args:
            seg_idx:  Segment index

        Returns:
            Segment ``seg_idx``
        """
        seg_start, seg_stop = self.bounds(seg_idx)
        return Segment(idx=seg_idx, start=seg_start, stop=seg_stop,
                       center=self.center(seg_idx),
                       n_frames=self._params.n_perseg, data=self[seg_idx])

    def __iter__(self) -> Generator[FloatArray, None, None]:
        for seg in self._segs.T:
            yield np.expand_dims(seg, 1)

    def __getitem__(self, key: int) -> FloatArray:
        out = self._segs[:, key]
        if out.ndim < 2:
            return np.expand_dims(out, 1)
        return out

    def __repr__(self) -> str:
        return f'Segments(params={self._params!s}, segs={self._segs!s})'

    def __str__(self) -> str:
        return f'<n_segs: {self.n_segs}, len_seg: {self._params.n_perseg}>'


class ArraySegmentation:
    # pylint: disable = R0903
    """Segementation"""
    def __init__(self, n_perseg: int, n_overlap: int, extend: bool = True,
                pad: bool = True) -> None:
        """Subdivide input array.

        Args:
            n_perseg:  Samples per segment
            n_overlap: Overlap in samples
            extend:    Extend a half window at start and end
            pad:       Pad extension
        """
        if n_perseg > 0:
            self.n_perseg = n_perseg
        else:
            msg = (f'Argument to ``n_perseg`` must be greater than '
                   f'zero.\nFound ``n_perseg`` = {n_perseg}.')
            raise ValueError(msg)

        if 0 < n_overlap < n_perseg:
            self.n_overlap = n_overlap
        else:
            msg = (f'Argument to ``n_overlap`` must be greater than '
                   f'zero and less then ``n_perseg``.\n Found '
                   f'``n_perseg`` = {self.n_perseg} and ``n_overlap`` '
                   f' = {n_overlap}.')
            raise ValueError(msg)

        self._extend = extend
        self._pad = pad
        self._ext_len = 0
        self._pad_len = 0

    def transform(self, data: FloatArray) -> Segments:
        """Apply segmentation.

        Input array must be either one-, or two-dimensional.
        If ``data`` is two-dimensional, it must be of shape
        ``(n_elements, 1)``.

        Args:
            data:  Input array

        Returns:
            ``Segments`` object
        """
        self._validate_data_shape(data)
        self._validate_nps(data.shape[0])
        n_frames = data.shape[0]
        step = self.n_perseg - self.n_overlap

        if self._extend:
            self._ext_len = self.n_perseg // 2

        if self._pad:
            self._pad_len = (-(n_frames-self.n_perseg) % step) % self.n_perseg

        data = np.pad(data.squeeze(), (self._ext_len, self._ext_len+self._pad_len))
        new_shape = data.shape[:-1] + ((data.shape[-1] - self.n_overlap) // step, self.n_perseg)
        # see https://github.com/PyCQA/pylint/issues/7981
        # pylint: disable = E1136
        new_strides = data.strides[:-1] + (step * data.strides[-1], data.strides[-1])
        segs = as_strided(data, new_shape, new_strides, writeable=False).T
        params = SegmentationParams(n_perseg=self.n_perseg, n_overlap=self.n_overlap,
                                    extend=self._extend, pad=self._pad)
        return Segments(params, segs)

    def _validate_nps(self, n_frames: int) -> None:
        if self.n_perseg > n_frames:
            msg = (f'Input data length ({n_frames}) incompatible with '
                    'parameter ``n_perseg`` = {self.n_perseg}. ``n_perseg`` '
                    'must be less then or equal to input data length.')
            raise ValueError(msg)

    def _validate_data_shape(self, data: FloatArray) -> None:
        if not 0 < data.ndim < 3:
            msg = (f'Input array must have one or two dimensions.\n'
                   f'Found ``data.shape`` = {data.shape}.')
        elif data.ndim == 2 and data.shape[1] != 1:
            msg = (f'Two-dimensional import arrays can only have one '
                   f'column.\nFound ``data.shape``= {data.shape}.')
        else:
            return None
        raise ValueError(msg)


class FileSegmentation:
    # pylint: disable = too-many-instance-attributes, too-many-arguments
    """Read segments from audio file."""
    def __init__(self, snd: AudioFile, n_perseg: int, n_overlap: int,
                 norm: bool = False, mono: bool = True,
                 expand: bool = True, dtype: str = 'float64') -> None:
        """Compute equal-sized segments.

        Args:
            snd:
            n_perseg:   Number of samples per segment
            n_overlap:  Size of segment overlap in samples
            norm:       Normalize each segment separately
            mono:       If ``True`` mixdown all channels
            expand:     Start segmentation at :math:`-n_perseg//2`
            dtype:      Dtype of output array
        """
        self._snd = snd
        self.n_perseg = n_perseg
        self.n_overlap = n_overlap
        self.expand = expand
        self.n_segs = int(np.ceil(self._snd.n_frames / n_overlap))
        if expand:
            self.n_segs += 1
            self.offset = -self.n_perseg // 2
        else:
            self.n_segs -= 1
            self.offset = 0
        self.step = self.n_perseg - self.n_overlap
        self.norm = norm
        self.mono = mono
        self.dtype = dtype

    def compute_bounds(self, seg_idx: int) -> tuple[int, int]:
        """Compute egment boundaries

        Compute segment boundaries in samples relative to zero from segment
        index ``seg_idx``.

        Args:
            seg_idx:    Index of requested segment

        Returns:
            Index of first and last sample in segment
        """
        if seg_idx < 0:
            raise IndexError('Expected positive integer for ``seg_idx``. '
                             f'Got {seg_idx}.')
        if seg_idx >= self.n_segs:
            raise IndexError(f'You requested segment {seg_idx}, but there '
                             f'are only {self.n_segs} segments.')
        start = seg_idx * self.n_overlap + self.offset
        return start, start + self.n_perseg

    def read_segment(self, seg_idx: int, norm: bool | None = None,
                     mono: bool | None = None, dtype: str | None = None
                     ) -> FloatArray:
        """Read single segement from file

        Args:
            seg_idx:    Segment index
            norm:       If `True`, normalize return array
            mono:       If `True`, downmix all channels
            dtype:      Dtype of return array

        Returns:
            Array segment data
        """
        norm = norm or self.norm
        mono = mono or self.mono
        dtype = dtype or self.dtype
        offset = seg_idx * self.n_overlap + self.offset
        return self._snd.read(self.n_perseg, offset, norm, mono, dtype)

    def loc(self, seg_idx: int, norm: bool | None = None,
            mono: bool | None = None, dtype: str | None = None) -> Segment:
        """Locate segment by index.

        Args:
            seg_idx:  Segment index
            norm:     If ``True``, normalize each segment separately
                      Falls back to ``self.norm``
            mono:     If ``True`` mixdown all channels
                      Falls back to ``self.mono``
            dtype:    Output dtype. Falls back to ``self.dtype``

        Returns:
            Segment number ``seg_idx``
        """
        seg_start, seg_stop = self.compute_bounds(seg_idx)
        data = self.read_segment(seg_idx, norm, mono, dtype)
        return Segment(idx=seg_idx, start=seg_start, stop=seg_stop,
                       center=self.n_perseg, n_frames=self._snd.fps, data=data)

    def __getitem__(self, key: int) -> Segment:
        return self.loc(key)

    def __iter__(self) -> Generator[Segment, None, None]:
        for i in range(self.n_segs):
            yield self.__getitem__(i)

    def iter_data(self) -> Generator[NDArray, None, None]:
        """Iterate over segment data"""
        for _ in range(self.n_segs):
            yield self._snd.read(self.n_perseg)

    def iter_bounds(self) -> Generator[tuple[int, int], None, None]:
        """Iterate over segment boundaries"""
        for i in range(self.n_segs):
            yield self.compute_bounds(i)
