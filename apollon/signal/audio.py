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
import scipy.io.wavfile as spw

from apollon.io import FileAccessControl
from apollon.tools import normalize
#from apollon import aplot as _aplot


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

    __slots__ = ['_sample_rate', '_signal', '_N', 'normalized']

    # Descriptor attribute
    file = FileAccessControl()

    def __init__(self, file_name, norm=True):
        '''Representation of an audio file.

        Params:
            file_name   (str)   Name of file.
            norm        (bool)  If True, signal will be normalized.

        Return:
            (AudioData) Object
        '''
        self.file = file_name

        self._sample_rate, self._signal = spw.read(file_name)
        self._N = len(self._signal)

        if self._signal.ndim != 1:
            if self._signal.shape[1] == 2:    # stereo ?
                self._signal = self._signal.sum(axis=1) / 2
            else:
                raise ValueError('Audio files with max. 2 channels, only.')

        self.normalized = norm
        if norm:
            self._normalize()


    def get_sr(self):
        '''Return sample rate.'''
        return self._sample_rate

    def get_data(self, n=None):
        '''Return the audio frames as ints.
        :param n:   (int) return only the first n frames (default = None)
        :return:    (array) frames
        '''
        return self._signal[:n]

    def plot(self, tickunit='seconds'):
        _aplot.signal(self, xaxis=tickunit)

    def _normalize(self):
        self._signal = self._signal / _np.max(_np.absolute(self._signal))
        self.normalized = True

    def __str__(self):
        return "<{}, Samples: {}, Sample rate: {}, Normalized: {}>" \
        .format(self.file.name,  self._N, self._sample_rate, self.normalized)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self._N

    def __getitem__(self, item):
        return self._signal[item]


def loadwav(path, norm=True):
    """Load a .wav file.

    Params:
        path    (str or fobject)
        norm    (bool) True if data should be normalized.

    Return:
        (int, ndarray)    sample rate and data.
    """
    if isinstance(path, str):
        return _AudioData(path, norm)
    elif isinstance(path, pathlib.Path):
        return _AudioData(str(path), norm)
    else:
        raise ValueError('`path` must be str or Path not {}'.format(type(path)))
