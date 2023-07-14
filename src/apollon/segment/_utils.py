# pylint: disable = C0114

import numpy as np

from apollon.types import FloatArray, IntArray
from apollon.signal.tools import zero_padding


def by_samples(arr: FloatArray, n_perseg: int, hop_size: int = 0) -> FloatArray:
    r"""Segment the input into :math:`n` segments of length ``n_perseg`` and move the
    window ``hop_size`` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    If ``hop_size`` is less than one, it is reset to ``n_perseg``.

    Overlap in percent is calculated as
    :math:`\frac{\text{hop_size}}{\text{n_perseg}} * 100`.

    Args:
        arr:        One-dimensional input array
        n_perseg:   Length of segments in samples
        hop_size:   Hop size in samples. If < 1, ``hop_size`` = ``n_perseg``.

    Returns:
        Two-dimensional array of segments.
    """
    if hop_size < 1:
        return _by_samples(arr, n_perseg)
    return _by_samples_with_hop(arr, n_perseg, hop_size)


def by_ms(arr: FloatArray, fps: int, ms_perseg: int, hop_size: int = 0) -> FloatArray:
    """Segment the input into n segments of length ``ms_perseg`` and move the
    window ``hop_size`` milliseconds.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    If ``hop_size`` is less than one, it is reset to ``n_perseg``.

    Overlap in percent is calculated as
    :math:`\frac{\text{hop_size}}{\text{n_perseg}} * 100`.

    Args:
        arr:        One-dimensional input array
        fps:        Sample rate in frames per second
        n_perseg:   Length of segments in milliseconds
        hop_size:   Hop size in milliseconds. If < 1, ``hop_size`` = ``n_perseg``.

    Returns:
        Two-dimensional array of segments
        """
    n_perseg = fps * ms_perseg // 1000
    hop_size = fps * hop_size // 1000
    return by_samples(arr, n_perseg, hop_size)



def by_onsets(arr: FloatArray, n_perseg: int, ons_idx: IntArray, off: int = 0
              ) -> FloatArray:
    """Split input ``arr`` into ``len(ons_idx)`` segments of length ``n_perseg``.

    Extraction windos start at ``ons_idx[i]`` + ``off``.

    Args:
        arr:        One-dimensional input array
        n_perseg:   Length of segments in samples
        ons_idx:    One-dimensional array of onset positions
        off:        Length of offset

    Returns:
        Two-dimensional array of shape ``(len(ons_idx), n_perseg)``.
    """
    n_ons = ons_idx.size
    out = np.empty((n_ons, n_perseg), dtype=arr.dtype)

    for i, idx in enumerate(ons_idx):
        pos = idx + off
        if pos < 0:
            pos = 0
        elif pos >= arr.size:
            pos = arr.size - 1

        if pos + n_perseg >= arr.size:
            buff = arr[pos:]
            out[i] = zero_padding(buff, n_perseg-buff.size)
        else:
            out[i] = arr[pos:pos+n_perseg]
    return out


def _by_samples(arr: FloatArray, n_perseg: int) -> FloatArray:
    """Split ``arr`` into segments of lenght ``n_perseg`` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    Args:
        arr:       One-dimensional input array
        n_perseg:  Length of segments in samples

    Returns:
        Two-dimensional array of segments
    """
    if not isinstance(n_perseg, int):
        raise TypeError('Param ``n_perchunk`` must be of type int.')

    if n_perseg < 1:
        raise ValueError('``n_perchunk`` out of range. '
                         'Expected 1 <= n_perchunk.')

    fit_size = int(np.ceil(arr.size / n_perseg) * n_perseg)
    n_ext = fit_size - arr.size
    arr = zero_padding(arr, n_ext)
    return arr.reshape(-1, n_perseg)


def _by_samples_with_hop(arr: FloatArray, n_perseg: int, hop_size: int) -> FloatArray:
    """Split ``arr`` into segments of lenght ``n_perseg`` samples. Move the
    extraction window ``hop_size`` samples.

    This function automatically applies zero padding for inputs that cannot be
    split evenly.

    Args:
        arr:       One-dimensional input array
        n_perseg:  Length of segments in samples
        hop_size:  Hop size in samples

    Returns:
        Two-dimensional array of segments
    """
    if not (isinstance(n_perseg, int) and isinstance(hop_size, int)):
        raise TypeError('Params must be of type int.')

    if not 1 < n_perseg <= arr.size:
        raise ValueError('n_perseg out of range. '
                         'Expected 1 < n_perseg <= len(arr).')

    if hop_size < 1:
        raise ValueError('hop_size out of range. Expected 1 < hop_size.')

    n_hops = (arr.size - n_perseg) // hop_size + 1
    n_segs = n_hops

    if (arr.size - n_perseg) % hop_size != 0 and n_perseg > hop_size:
        n_segs += 1

    fit_size = hop_size * n_hops + n_perseg
    n_ext = fit_size - arr.size
    arr = zero_padding(arr, n_ext)

    out = np.empty((n_segs, n_perseg), dtype=arr.dtype)
    for i in range(n_segs):
        off = i * hop_size
        out[i] = arr[off:off+n_perseg]
    return out
