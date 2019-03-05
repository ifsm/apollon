"""apollon/segment.py
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

    if n_perchunk < 1:
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
    """Segment the input into n segments of length n_persegs and move the
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


def by_ms(signal_obj, len_of_chunks_in_ms, padding=True, sr=44100):
    data, obj_sr = _get_audio_data(signal_obj)
    if obj_sr is None:
        sample_rate = sr
    else:
        sample_rate = obj_sr
    len_of_chunks_in_samples = int(sample_rate * len_of_chunks_in_ms / 1000)
    return by_samples(signal_obj, len_of_chunks_in_samples, padding=padding)


def by_ms_with_hop(signal_obj, len_of_chunks_in_ms, hop_size_in_ms, sr=44100):
    data, obj_sr = _get_audio_data(signal_obj)
    if obj_sr is None:
        sample_rate = sr
    else:
        sample_rate = obj_sr
    len_of_chunks_in_samples = int(sample_rate * len_of_chunks_in_ms / 1000)
    hop_size_in_samples = int(sample_rate * hop_size_in_ms / 1000)
    return by_samples_with_hop(signal_obj, len_of_chunks_in_samples,
                               hop_size_in_samples)


def from_onsets(signal_obj, odx, clen, energy=False):
    '''Constructs AudioChunk object based on onsets.

    :param signal_obj:      (AudioData) audio signal
    :param odx:             (array-like) of onset indices
    :param clen:            (int) length of chunks given in samples

    :return:                (AudioChunks)
    '''
    data, sr = signal_obj.data, signal_obj.fs
    bounds = [(i, i+clen) for i in odx]

    padding = False
    if bounds[-1][1] >= len(data):
        n = bounds[-1][1] - len(data)
        tools.zero_padding(data, n)
        padding = n

    return _AudioChunks(data, len(bounds), clen, bounds, sr, padding)


def from_onsets_shift_back(signal_obj, odx, clen, energy=False):
    '''Constructs AudioChunk object based on onsets.

    :param signal_obj:      (AudioData) audio signal
    :param odx:             (array-like) of onset indices
    :param clen:            (int) length of chunks given in samples

    :return:                (AudioChunks)
    '''
    data, sr = signal_obj.data, signal_obj.fs
    shift = signal_obj.get_sr() // 100
    bounds = [(i-shift, i+(clen-shift)) for i in odx]

    padding = False
    if bounds[-1][1] >= len(data):
        n = bounds[-1][1] - len(data)
        tools.zero_padding(data, n)
        padding = n

    return _AudioChunks(data, len(bounds), clen, bounds, sr, padding)


def _get_audio_data(signal_obj):
    '''Returns data and sample rate from an arbitrary array-like or
        AudioData object.

        :param signal_obj:   (numerical array-like/AudioData)
        :return:             (array) signal, (int) samplerate
    '''
    if isinstance(signal_obj, _AudioData):
        data = signal_obj.data
        sr = signal_obj.fs
    else:
        if signal_obj.dtype.type is _np.str_:
            raise NotAudioDataError('Data contains elements of type {}'
                                    .format(signal_obj.dtype.type))
        data = _np.atleast_1d(signal_obj)
        sr = None

    return data, sr


if __name__ == 'main':
    print('Import module to use segmentation functionality.')
