"""
Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net
"""
from dataclasses import dataclass
from typing import ClassVar, Generator, Tuple, Union

import numpy as _np
from numpy.lib.stride_tricks import as_strided

from . audio import AudioFile
from . container import Params
from . signal.tools import zero_padding as _zero_padding
from . types import Array, Schema


@dataclass
class LazySegmentParams:
    """Encapsulates segmentation parameters."""
    n_perseg: int
    n_overlap: int
    norm: bool = False
    mono: bool = True
    expand: bool = True
    dtype: str = 'float64'


SEGMENTATION_PARAMS = {
    "type": "object",
    "properties": {
        "n_perseg": {"type": "integer"},
        "n_overlap": {"type": "integer"},
        "extend": {"anyOf": [{"type": "boolean"}, {"type": "integer"}]},
        "pad": {"anyOf": [{"type": "boolean"}, {"type": "integer"}]}
    }
}


@dataclass
class SegmentationParams(Params):
    """Parameters for Segmentation."""
    _schema: ClassVar[Schema] = SEGMENTATION_PARAMS
    n_perseg: int = 512
    n_overlap: int = 256
    extend: Union[bool, int] = True
    pad: Union[bool, int] = True


@dataclass
class Segment:
    """Encapsulates audio segment data and meta data."""
    idx: int
    start: int
    stop: int
    center: int
    n_frames: int
    data: _np.ndarray


class Segments:
    """Segement"""
    def __init__(self, params: SegmentationParams, segs: _np.ndarray) -> None:
        self._segs = segs
        self._params = params
        if self._params.extend:
            self._offset = 0
        else:
            self._offset = self._params.n_perseg // 2

    @property
    def data(self) -> Array:
        """Return the raw segment data array."""
        return self._segs

    @property
    def n_segs(self) -> int:
       return self._segs.shape[1]

    @property
    def n_perseg(self) -> int:
        return self._params.n_perseg

    @property
    def n_overlap(self) -> int:
        return self._params.n_overlap

    @property
    def step(self) -> int:
        return self._params.n_perseg - self._params.n_overlap

    @property
    def params(self) -> SegmentationParams:
        """Parameter set used to compute this instance."""
        return self._params

    def center(self, seg_idx) -> int:
        """Return the center of segment ``seg_idx`` as frame number
        of the original signal.

        Args:
            seg_indx:  Segment index.

        Returns:
            Center frame index.
        """
        if not (0 <= seg_idx < self.n_segs):
            raise IndexError('Requested index out of range.')
        return seg_idx * self.step + self._offset

    def bounds(self, seg_idx) -> Tuple[int, int]:
        """Return the frame numbers of the lower and upper bound
        of segment ``seg_idx``. Lower bound index is inclusive,
        upper bound index is exclusive.

        Args:
            seg_idx:  Segment index.

        Returns:
            Lower and upper bound frame index.
        """
        if not (0 <= seg_idx < self.n_segs):
            raise IndexError('Requested index out of range.')
        lob = self.center(seg_idx) - self._params.n_perseg // 2
        upb = lob + self._params.n_perseg
        return lob, upb

    def get(self, seg_idx) -> Segment:
        """Retrun segment ``seg_idx`` wrapped in an ``Segment`` object.

        Args:
            seg_idx:  Segment index.

        Returns:
            Segment ``seg_idx``.
        """
        return Segment(seg_idx, *self.bounds(seg_idx), self.center(seg_idx),
                       self._params.n_perseg, self[seg_idx])

    def __iter__(self) -> Generator[_np.ndarray, None, None]:
        for seg in self._segs.T:
            yield _np.expand_dims(seg, 1)

    def __getitem__(self, key) -> _np.ndarray:
        out = self._segs[:, key]
        if out.ndim < 2:
            return _np.expand_dims(out, 1)
        return out

    def __repr__(self) -> str:
        return f'Segments(params={self._params!s}, segs={self._segs!s})'

    def __str__(self) -> str:
        return f'<n_segs: {self.n_segs}, len_seg: {self._params.n_perseg}>'


class Segmentation:
    """Segementation"""
    def __init__(self, n_perseg: int, n_overlap: int, extend: bool = True,
                pad: bool = True) -> None:
        """Subdivide input array.

        Args:
            n_perseg:  Samples per segment.
            n_overlap: Overlap in samples.
            extend:    Extend a half window at start and end.
            pad:       Pad extension.
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

    def transform(self, data: _np.ndarray) -> Segments:
        """Apply segmentation.

        Input array must be either one-, or two-dimensional.
        If ``data`` is two-dimensional, it must be of shape
        (n_elements, 1).

        Args:
            data:  Input array.

        Returns:
            ``Segments`` object.
        """
        self._validate_data_shape(data)
        self._validate_nps(data.shape[0])
        n_frames = data.shape[0]
        step = self.n_perseg - self.n_overlap

        if self._extend:
            self._ext_len = self.n_perseg // 2

        if self._pad:
            self._pad_len = (-(n_frames-self.n_perseg) % step) % self.n_perseg

        data = _np.pad(data.squeeze(), (self._ext_len, self._ext_len+self._pad_len))
        new_shape = data.shape[:-1] + ((data.shape[-1] - self.n_overlap) // step, self.n_perseg)
        new_strides = data.strides[:-1] + (step * data.strides[-1], data.strides[-1])
        segs = as_strided(data, new_shape, new_strides, writeable=False).T
        params = SegmentationParams(self.n_perseg, self.n_overlap,
                                    self._extend, self._pad)
        return Segments(params, segs)

    def _validate_nps(self, n_frames: int) -> None:
        if self.n_perseg > n_frames:
            msg = (f'Input data length ({n_frames}) incompatible with '
                    'parameter ``n_perseg`` = {self.n_perseg}. ``n_perseg`` '
                    'must be less then or equal to input data length.')
            raise ValueError(msg)

    def _validate_data_shape(self, data: _np.ndarray) -> None:
        if not (0 < data.ndim < 3):
            msg = (f'Input array must have one or two dimensions.\n'
                   f'Found ``data.shape`` = {data.shape}.')
        elif data.ndim == 2 and data.shape[1] != 1:
            msg = (f'Two-dimensional import arrays can only have one '
                   f'column.\nFound ``data.shape``= {data.shape}.')
        else:
            return None
        raise ValueError(msg)


class LazySegments:
    """Read segments from audio file."""
    def __init__(self, snd: AudioFile, n_perseg: int, n_overlap: int,
                 norm: bool = False, mono: bool = True,
                 expand: bool = True, dtype: str = 'float64') -> None:
        """Compute equal-sized segments.

        Args:
            snd:
            n_perseg:   Number of samples per segment.
            n_overlap:  Size of segment overlap in samples.
            norm:       Normalize each segment separately.
            mono:       If ``True`` mixdown all channels.
            expand:     Start segmentation at -n_perseg//2.
            dtype:      Dtype of output array.
        """
        self._snd = snd
        self.n_perseg = n_perseg
        self.n_overlap = n_overlap
        self.expand = expand
        self.n_segs = int(_np.ceil(self._snd.n_frames / n_overlap))
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

    def compute_bounds(self, seg_idx):
        if seg_idx < 0:
            raise IndexError('Expected positive integer for ``seg_idx``. '
                             f'Got {seg_idx}.')
        if seg_idx >= self.n_segs:
            raise IndexError(f'You requested segment {seg_idx}, but there '
                             f'are only {self.n_segs} segments.')
        start = seg_idx * self.n_overlap + self.offset
        return start, start + self.n_perseg

    def read_segment(self, seg_idx: int, norm: bool = None,
                     mono: bool = None, dtype: str = None):
        norm = norm or self.norm
        mono = mono or self.mono
        dtype = dtype or self.dtype
        offset = seg_idx * self.n_overlap + self.offset
        return self._snd.read(self.n_perseg, offset, norm, mono, dtype)

    def loc(self, seg_idx: int, norm: bool = None,
            mono: bool = None, dtype: str = None) -> Segment:
        """Locate segment by index.

        Args:
            seg_idx:  Segment index.
            norm:     If ``True``, normalize each segment separately.
                      Falls back to ``self.norm``.
            mono:     If ``True`` mixdown all channels.
                      Falls back to ``self.mono``.
            dtype:    Output dtype. Falls back to ``self.dtype``.

        Returns:
            Segment number ``seg_idx``.
        """
        start, stop = self.compute_bounds(seg_idx)
        data = self.read_segment(seg_idx, norm, mono, dtype)
        return Segment(seg_idx, start, stop, self.n_perseg,
                       self._snd.fps, data)

    def __getitem__(self, key):
        return self.loc(key)

    def __iter__(self):
        for i in range(self.n_segs):
            yield self.__getitem__(i)

    def iter_data(self):

        for i in range(self.n_segs):
            yield self._snd.read(self.n_perseg)

    def iter_bounds(self):
        for i in range(self.n_segs):
            yield self.compute_bounds(i)


def _by_samples(x: Array, n_perseg: int) -> Array:
    """Split ``x`` into segments of lenght ``n_perseg`` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    Args:
        x:         One-dimensional input array.
        n_perseg:  Length of segments in samples.

    Returns:
        Two-dimensional array of segments.
    """
    if not isinstance(n_perseg, int):
        raise TypeError('Param ``n_perchunk`` must be of type int.')

    if n_perseg < 1:
        raise ValueError('``n_perchunk`` out of range. '
                         'Expected 1 <= n_perchunk.')

    fit_size = int(_np.ceil(x.size / n_perseg) * n_perseg)
    n_ext = fit_size - x.size
    x = _zero_padding(x, n_ext)
    return x.reshape(-1, n_perseg)


def _by_samples_with_hop(x: Array, n_perseg: int, hop_size: int) -> Array:
    """Split `x` into segments of lenght `n_perseg` samples. Move the
    extraction window `hop_size` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    Args:
        x:         One-dimensional input array.
        n_perseg:  Length of segments in samples.
        hop_size:  Hop size in samples

    Returns:
        Two-dimensional array of segments.
    """
    if not (isinstance(n_perseg, int) and isinstance(hop_size, int)):
        raise TypeError('Params must be of type int.')

    if not 1 < n_perseg <= x.size:
        raise ValueError('n_perseg out of range. '
                         'Expected 1 < n_perseg <= len(x).')

    if hop_size < 1:
        raise ValueError('hop_size out of range. Expected 1 < hop_size.')

    n_hops = (x.size - n_perseg) // hop_size + 1
    n_segs = n_hops

    if (x.size - n_perseg) % hop_size != 0 and n_perseg > hop_size:
        n_segs += 1

    fit_size = hop_size * n_hops + n_perseg
    n_ext = fit_size - x.size
    x = _zero_padding(x, n_ext)

    out = _np.empty((n_segs, n_perseg), dtype=x.dtype)
    for i in range(n_segs):
        off = i * hop_size
        out[i] = x[off:off+n_perseg]
    return out


def by_samples(x: Array, n_perseg: int, hop_size: int = 0) -> Array:
    """Segment the input into n segments of length n_perseg and move the
    window `hop_size` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    If `hop_size` is less than one, it is reset to `n_perseg`.

    Overlap in percent is calculated as ov = hop_size / n_perseg * 100.

    Args:
        x           One-dimensional input array.
        n_perseg    Length of segments in samples.
        hop_size    Hop size in samples. If < 1, hop_size = n_perseg.

    Returns:
        Two-dimensional array of segments.
    """
    if hop_size < 1:
        return _by_samples(x, n_perseg)
    else:
        return _by_samples_with_hop(x, n_perseg, hop_size)


def by_ms(x: Array, fps: int, ms_perseg: int, hop_size: int = 0) -> Array:
    """Segment the input into n segments of length ms_perseg and move the
    window `hop_size` milliseconds.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    If `hop_size` is less than one, it is reset to `n_perseg`.

    Overlap in percent is calculated as ov = hop_size / n_perseg * 100.

    Args:
        x           One-dimensional input array.
        fs          Sampling frequency.
        n_perseg    Length of segments in milliseconds.
        hop_size    Hop size in milliseconds. If < 1, hop_size = n_perseg.

    Returns:
        Two-dimensional array of segments.
        """
    n_perseg = fps * ms_perseg // 1000
    hop_size = fps * hop_size // 1000
    return by_samples(x, n_perseg, hop_size)


def by_onsets(x: Array, n_perseg: int, ons_idx: Array, off: int = 0
              ) -> Array:
    """Split input `x` into len(ons_idx) segments of length `n_perseg`.

    Extraction windos start at `ons_idx[i]` + `off`.

    Args:
        x           One-dimensional input array.
        n_perseg    Length of segments in samples.
        ons_idx     One-dimensional array of onset positions.
        off         Length of offset.

    Returns:
        Two-dimensional array of shape (len(ons_idx), n_perseg).
    """
    n_ons = ons_idx.size
    out = _np.empty((n_ons, n_perseg), dtype=x.dtype)

    for i, idx in enumerate(ons_idx):
        pos = idx + off
        if pos < 0:
            pos = 0
        elif pos >= x.size:
            pos = x.size - 1

        if pos + n_perseg >= x.size:
            buff = x[pos:]
            out[i] = _zero_padding(buff, n_perseg-buff.size)
        else:
            out[i] = x[pos:pos+n_perseg]
    return out
