#!/usr/bin/python
# -*- coding: utf-8 -*-


import math as _math
from collections import OrderedDict

import numpy as _np

from apollon.decorators import isAudioChunk
from apollon.fractal.phasespace import PseudoPhaseSpace
from apollon.signal.audio import _AudioChunks
from apollon.signal.spectral import fft
from apollon.tools import scale


__author__ = 'Michael Blaß'


@isAudioChunk
def maxamp(chunk, **kwargs):
    '''Extract maximal amplitude from chunk.'''
    return _np.max(_np.absolute(chunk))


@isAudioChunk
def meanamp(chunk, **kwargs):
    '''Extract mean amplitude from chunk.'''
    return _np.sum(chunk) / len(chunk)


@isAudioChunk
def stdamp(chunk, **kwargs):
    '''Extract standard deviation from chunk.'''
    return _np.std(chunk)


@isAudioChunk
def energy(chunk, **kwargs):
    '''Extract linear energy from chunk.'''
    return _np.sum(_np.square(chunk))


# @isAudioChunk
def entropy(chunk, theta=2, bins=10, window=None, **kwargs):
    '''Shannon Entropy in [nat], normalized by log(N)'''
    if window:
        pps = PseudoPhaseSpace(chunk * window(len(chunk)), theta, bins)
    else:
        pps = PseudoPhaseSpace(chunk, theta, bins)
    return pps.get_entropy()


@isAudioChunk
def rms_energy(chunk, **kwargs):
    """Extract rms energy from chunk."""
    return _math.sqrt(_np.sum(_np.square(chunk)) / len(chunk))


def spectral_centroid(chunks, window=None, sr=None, **kwargs):
    """Extract spectral centroid from chunks.

    Params:
        window      (function) window function.
        sr          (int)   sample rate i Hz.

    return    (numpy ndarray) of spc values.
    """
    if isinstance(chunks, _AudioChunks):
        out = _np.zeros(len(chunks))

        for i, chunk in enumerate(chunks):
            X = fft(chunk, sr=sr, window=window)
            out[i] = X.centroid()
        return out

    else:
        spc = fft(chunks, sr=sr, window=window)
        return spc.centroid()


available_features = OrderedDict({'maxamp': maxamp,
                                  'meanamp': meanamp,
                                  'stdamp': stdamp,
                                  'entropy': entropy,
                                  'energy': energy,
                                  'rms': rms_energy,
                                  'spc': spectral_centroid,
                                  })
