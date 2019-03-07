"""
Copyright 2019 Michael Blaß
<michael.blass@uni-hamburg.de>
"""

import numpy as _np

from . signal.tools import zero_padding as _zero_padding
from . types import Array as _Array


def _by_samples(x: _Array, n_perseg: int) -> _Array:
    """Split `x` into segments of lenght `n_perseg` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    Args:
        x        (np.ndarray)    One-dimensional input array.
        n_perseg (int)           Length of segments in samples.

    Returns:
        (np.ndarray)    Two-dimensional array of segments.
    """
    if not isinstance(n_perseg, int):
        raise TypeError('Param `n_perchunk` must be of type int.')

    if n_perseg < 1:
        raise ValueError('`n_perchunk` out of range. Expected 1 <= n_perchunk.')

    fit_size = int(_np.ceil(x.size / n_perchunk) * n_perseg)
    n_ext = fit_size - x.size
    x = _zero_padding(x, n_ext)

    return x.reshape(-1, n_perseg)


def _by_samples_with_hop(x: _Array, n_perseg: int, hop_size: int) -> _Array:
    """Split `x` into segments of lenght `n_perseg` samples. Move the extraction
    window `hop_size` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    Args:
        x        (np.ndarray)    One-dimensional input array.
        n_perseg (int)           Length of segments in samples.
        hop_size (int)           Hop size in samples

    Returns:
        (np.ndarray)    Two-dimensional array of segments.
    """
    if not (isinstance(n_perseg, int) and isinstance(hop_size, int)):
        raise TypeError('Params must be of type int.')

    if not 1 < n_perseg <= x.size:
        raise ValueError('n_perseg out of range. Expected 1 < n_perseg <= len(x).')

    if hop_size < 1:
        raise ValueError('hop_size out of range. Expected 1 < hop_size.')

    n_hops = (x.size - n_perseg) // hop_size + 1
    n_segs = n_hops

    if (x.size - n_perseg) % hop_size != 0:
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
        (np.ndarray)    Two-dimensional array of segments.
    """
    if hop_size < 1:
        return _by_samples(x, n_perseg)
    else:
        return _by_samples_with_hop(x, n_perseg, hop_size)


def by_ms(x: _Array, fs: int, ms_perseg: int, hop_size: int = 0) -> _Array:
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
        (np.ndarray)    Two-dimensional array of segments.
        """
    n_perseg = fs * ms_perseg // 1000
    hop_size = fs * hop_size // 1000

    return by_samples(x, n_perseg, hop_size)


def by_onsets(x: _Array, n_perseg: int, ons_idx: _Array, off: int = 0) -> _Array:
    """Split input `x` into len(ons_idx) segments of length `n_perseg`.

    Extraction windos start at `ons_idx[i]` + `off`.

    Args:
        x        (np.ndarray)    One-dimensional input array.
        n_perseg (int)           Length of segments in samples.
        ons_idx  (np.ndarray)    One-dimensional array of onset positions.
        off      (int)           Length of offset.

    Returns:
        (np.ndarray)    Two-dimensional array of shape (len(ons_idx), n_perseg).
    """
    n_ons = ons_idx.size
    out = _np.empty((n_ons, n_perseg))

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
