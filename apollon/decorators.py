#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Michael Blaß'

from apollon.signal.audio import _AudioChunks
import numpy as _np
import matplotlib.pyplot as _plt


def isAudioChunk(func):
    def wrapper(inp, **kwargs):
        if isinstance(inp, _AudioChunks):
            return _np.array([func(i, **kwargs) for i in inp])
        else:
            return func(inp)
    return wrapper

def isAudioChunkSPC(func):
    def wrapper(inp, **kwargs):
        if isinstance(inp, _AudioChunks):
            return _np.array([func(i, **kwargs) for i in inp])
        else:
            return func(inp)
    return wrapper

def timing(func):
    # TODO: implement timing function
    raise NotImplementedError
