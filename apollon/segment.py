#!/usr/bin/python
# -*- coding: utf-8 -*-

"""apollon/segment.py

(c) Michael Blaß, 2016

This module encapsulates a bunch of function for simple audio segmentation.
A given AudioData signal is segmented to chunks by means of either samples or
seconds, each with or without hop size. There is no need to instantiate Split()
since its member functions are all static.
"""


import numpy as _np

from apollon.exceptions import NotAudioDataError
from apollon.signal import tools
from apollon.signal.audio import _AudioChunks
from apollon.signal.audio import _AudioData


def by_samples(signal_obj, len_of_chunks_in_samples, padding=True):
    """A signal of raw audio data is segmented by means of samples. Since there
    is no zero padding applied by default, the last segment may be shorter then
    the other if len(signal) % len_of_parts != 0.

    :param signal_obj:               (AudioData) audio object
    :param len_of_chunks_in_samples: (int) length of each part given in samples
    :param padding:                  (bool) apply zero padding if true
    :return:    AudiChunks object
    """
    len_sig = len(signal_obj)

    # security
    if len_of_chunks_in_samples < 1 or len_of_chunks_in_samples >= len_sig:
        raise ValueError('Length must be: 1 <= part_length >= len(signal).')

    if type(len_of_chunks_in_samples) is not int:
        raise ValueError('Length must be an integer > 1.')

    # get data
    data, sr = _get_audio_data(signal_obj)

    # calculate number of parts
    if len_sig % len_of_chunks_in_samples:
        n_parts = int(len_sig / len_of_chunks_in_samples) + 1
    else:
        n_parts = int(len_sig / len_of_chunks_in_samples)

    # set up array for parts and the corresponding limits
    parts = _np.zeros([n_parts, len_of_chunks_in_samples], dtype=data.dtype)
    limits = []
    parts_counter = 0

    # zero-padding
    if padding:
        padding = parts.size - len_sig
        data = tools.zero_padding(data, padding)
        len_sig = len(data)

    # calculate parts
    for i in range(0, len_sig, len_of_chunks_in_samples):
        if (i + len_of_chunks_in_samples) <= len_sig:
            limits.append([i, i+len_of_chunks_in_samples])
        else:
            limits.append([i, len_sig])
        parts_counter += 1

    return _AudioChunks(data, n_parts, len_of_chunks_in_samples,
                             _np.array(limits), sr, padding)


def by_samples_with_hop(signal_obj, len_of_chunks_in_samples,
                        hop_size_in_samples):
    '''Segments a signal into subchunks given a chunk length and hop
        size. Using the formula (len_chunk + hop_size * x) = len_sig
        the number of chunks as well as the padding can be calculated
        easily.

        Params:
            len_of_chunks_in_samples:    (int) chunk length in samples
            hop_size_in_samples:         (int) hop size in samples
    '''
    len_sig = len(signal_obj)

    # security
    if not (isinstance(len_of_chunks_in_samples, int) and
            isinstance(hop_size_in_samples, int)):
        raise ValueError('Length and hop size have to be ints > 1.')

    if not (1 <= len_of_chunks_in_samples <= len_sig):
        raise ValueError('Length have to be: 1 <= part_length >= len(signal).')

    if hop_size_in_samples < 1:
        raise ValueError('Hop sizes have to be ints > 0')

    # get the data
    data, sr = _get_audio_data(signal_obj)

    # calculate number of chunks and len of padding
    if (len_sig - len_of_chunks_in_samples) % hop_size_in_samples:
        nhchunks = ((len_sig - len_of_chunks_in_samples) //
                    hop_size_in_samples + 1)
        nchunks = nhchunks + 1
        padding = ((len_of_chunks_in_samples + hop_size_in_samples *
                    nhchunks) - len_sig)
        data = tools.zero_padding(data, padding)
    else:
        nchunks = ((len_sig - len_of_chunks_in_samples) //
                   hop_size_in_samples + 1)
        padding = False

    # set up array of chunk bounds
    bounds = _np.zeros((nchunks, 2), dtype=int)
    for i in range(nchunks):
        lb = i * hop_size_in_samples
        ub = lb + len_of_chunks_in_samples
        bounds[i] = [lb, ub]

    return _AudioChunks(data, nchunks, len_of_chunks_in_samples,
                             bounds, sr, padding)


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
    data, sr = signal_obj.get_data(), signal_obj.get_sr()
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
    data, sr = signal_obj.get_data(), signal_obj.get_sr()
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
        data = signal_obj.get_data()
        sr = signal_obj.get_sr()
    else:
        if signal_obj.dtype.type is _np.str_:
            raise NotAudioDataError('Data contains elements of type {}'
                                    .format(signal_obj.dtype.type))
        data = _np.atleast_1d(signal_obj)
        sr = None

    return data, sr


if __name__ == 'main':
    print('Import module to use segmentation functionality.')
