#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Classes:
    _AudioChunks
    _AudioData

Functions:
    loadwav         Load .wav file.
"""


import numpy as _np
import pathlib
import soundfile as sf

from apollon.io import FileAccessControl
from apollon.signal.tools import normalize


__author__ = 'Michael Blaß'


class _AudioChunks:
    '''    Representation of chunked audio data.

    Adds audio chunking functionality to an object.
    This class makes it easy to deal with evenly segmented audio signals.
    It also contains the original audio data as AudiData object. Thus, the
    original AudiData object may be deleted after creating an AudioChunks
    object.

    Methods:
        get_parts(self)
        iter_index(self)
        iter_limits(self)
        iter_parts(self)
        get_limits(self)
        get_sr(self)
    '''
    def __init__(self, _signal, _nchunks, _lchunks, _limits,
                 _sample_rate, _padding=False):
        '''        :param _signal:        (AudioData)  Audiodata object
        :param _nchunks:       (int)        Number of chunks
        :param _lchunks:       (int)        Length of each chunk
        :param _limits:        (array)      Indices of first and last frame
        :param _sample_rate:   (int)        Sample rate chunks
        :param _padding:       (bool)       True if the signal was zero-padded
        '''
        self._signal = _signal
        self._Nchunks = _nchunks
        self._lchunks = _lchunks
        self._limits = _limits
        self._sample_rate = _sample_rate
        self._padding = _padding

    def get_chunks(self):
        '''Return chunks as nested array.'''
        return _np.array([self._signal[start:stop]
                         for start, stop in self._limits])

    def get_chunk_len(self):
        return self._lchunks

    def iter_index(self):
        for i in range(self._Nchunks):
            yield i

    def iter_chunks(self):
        for start, stop in self._limits:
            out = _np.zeros(self._lchunks)
            data = self._signal[start:stop]
            out[:len(data)] = data
            yield out

    def get_limits(self):
        return self._limits

    def get_sr(self):
        return self._sample_rate

    def __str__(self):
        if self._padding:
            return '<AudioChunks object, N: {}, Len of each: {}, zero padding of len {}>'. \
                format(self._Nchunks, self._lchunks, self._padding)
        else:
            return '<AudioChunks object, N: {}, Len of each: {}, No padding>'.format(self._Nchunks, self._lchunks)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self._Nchunks

    def __getitem__(self, item):
        if isinstance(item, int):
            start, stop = self._limits[item]
            out = _np.zeros(self._lchunks)
            data = self._signal[start:stop]
            out[:len(data)] = data
            return out
        else:
            out = _np.zeros((len(item), self._lchunks))
            for i, val in enumerate(item):
                start, stop = self._limits[val]
                x = self._signal[start:stop]
                out[i][:len(x)] = x
            return out

    def __iter__(self):
        return self.iter_chunks()


class _AudioData:

    __slots__ = ['_fs', '_data']

    # Descriptor attribute
    file = WaveFileAccessControl()

    def __init__(self, file_name, norm=True):
        """Representation of an audio file.

        Params:
            file_name   (str)   Name of file.
            norm        (bool)  If True, signal will be normalized.

        Return:
            (AudioData) Object
        """
        self.file = file_name

        self._data, self._fs = sf.read(file_name, dtype='float64')

        if self._data.ndim == 2:
            self._data = self._data.sum(axis=1) / 2

        if norm:
            self._data = normalize(self._data)

    @property
    def fs(self):
        """Return sample rate."""
        return self._fs

    @property
    def data(self, n=None):
        """Return the audio frames as ints.

        Params:
            n: (int)    Return only the first n frames (default = None)

        Return:
            (np.ndarray) frames
        """
        return self._data[:n]

    def plot(self, tickunit='seconds'):
        _aplot.signal(self, xaxis=tickunit)

    def __str__(self):
        return "<{}, fs: {}, N: {}>" \
        .format(self.file.name,  self.fs, len(self))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.data.size

    def __getitem__(self, item):
        return self._data[item]


def loadwav(path, norm=True):
    """Load a .wav file.

    Params:
        path    (str or fobject)
        norm    (bool) True if data should be normalized.

    Return:
        (_AudioData) object.
    """
    return _AudioData(path, norm)
