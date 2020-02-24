"""
Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net
"""
from dataclasses import dataclass
import numpy as _np

from . audio import AudioFile
from . signal.tools import zero_padding as _zero_padding
from . types import Array as _Array


@dataclass
class SegmentParams:
    """Encapsulates segmentation parameters."""
    n_perseg: int
    n_overlap: int
    norm: bool = False
    mono: bool = True
    expand: bool = True
    dtype: str = 'float64'


@dataclass
class Segment:
    """Encapsulates audio segment data and meta data."""
    idx: int
    start: int
    stop: int
    n_frames: int
    fps: int
    data: _np.ndarray


class Segments:
    """Segementation"""
    def __init__(self, n_perseg: int, n_overlap: int, extend=True,
                 pad=True) -> None:
        """Divide a one-dimensional array into possibly overlapping segments.

        Input must be one-dimensional.

        Args:
            n_perseg:  Samples per segment.
            n_overlap: Overlap in samples.
            extend:    Extend a half window at start and end.
            pad:       Pad extension.
        """
        self.n_perseg = n_perseg
        self.n_overlap = n_overlap
        self.step = self.n_perseg - self.n_overlap
        self._extend = True
        self._pad = True
        self._ext_len = 0
        self._pad_len = 0

    def transform(self, data: _np.ndarray) -> _np.ndarray:
        """Apply segmentation."""
        n_sig = data.shape[0]

        if self._extend:
            self._ext_len = self.n_perseg // 2

        if self._pad:
            self._pad_len = (-(n_sig-self.n_perseg) % self.step) % self.n_perseg

        data = _np.pad(data, (self._ext_len, self._ext_len+self._pad_len))

        step = self.n_perseg - self.n_overlap
        new_shape = data.shape[:-1] + ((data.shape[-1] - self.n_overlap) // self.step, self.n_perseg)
        new_strides = data.strides[:-1] + (step * data.strides[-1], data.strides[-1])
        return _np.lib.stride_tricks.as_strided(data, new_shape, new_strides, writeable=False).T


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


def _by_samples(x: _Array, n_perseg: int) -> _Array:
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


def _by_samples_with_hop(x: _Array, n_perseg: int, hop_size: int) -> _Array:
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


def by_samples(x: _Array, n_perseg: int, hop_size: int = 0) -> _Array:
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


def by_ms(x: _Array, fps: int, ms_perseg: int, hop_size: int = 0) -> _Array:
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


def by_onsets(x: _Array, n_perseg: int, ons_idx: _Array, off: int = 0
              ) -> _Array:
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
