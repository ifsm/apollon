#!/usr/bin/python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as _plt
import numpy as _np
from scipy.signal import argrelmax, hilbert

from apollon import segment
from apollon import extract
from apollon import aplot as _aplot


def detect(sig, theta=10, boxes=50, chunk_len=16, hop_size=8, order=15, hilb=True, lb=True):
    '''Detects note onsets in percussive musical signals using weighted information entropy minimization.
    :param sig:         (AudioData) audio signal
    :param theta:       (int) order of delay in samples
    :param boxes:       (int) number of boxes per axis
    :param chunk_len:   (int) len of detection chunks in ms
    :param hop_size:    (int) extraction window displacement in ms
    :param order:       (int) comparison area per side in samples
    :param hilb:        (bool) calculate the onsets from the hilbert transform of the signal
    :param hmm:         (bool) use HMM to correctliy idetify onsets
    :return:            (tuple) array of indices, list of entropy values
    '''
    chunks = segment.by_ms_with_hop(sig, chunk_len, hop_size)

    H = _np.array([extract.entropy(i, theta, boxes) for i in chunks])
    rgy = _np.array(extract.energy(chunks))
    if hilb:
        H = hilbert(H).imag *(-1)

    w = abs(H) * rgy
    odf = argrelmax(w, order=order)[0]
    if lb:
        onsets_idx = [i[0] for i in chunks.get_limits()[odf]]
    else:
        onsets_idx = [(i+j)//2 for (i, j) in chunks.get_limits()[odf]]


    return OnsetResult(theta, boxes, chunk_len, chunks.get_limits(), hop_size, order, w, onsets_idx, H, rgy)


class OnsetResult:
    '''Encapsulate the results of an onset detection.'''
    __slots__ = ['theta', 'boxes', 'chunk_len', 'limits', 'hop_size', 'order', 'odf', 'odx', 'H', 'rgy']

    def __init__(self, theta, boxes, chunk_len, chunk_limits, hop_size, order, odf, odx, H, rgy):

        self.theta = theta
        self.boxes = boxes
        self.chunk_len = chunk_len
        self.limits = chunk_limits
        self.hop_size = hop_size
        self.order = order
        self.odf = odf
        self.odx = odx
        self.H = H
        self.rgy = rgy

    def plot(self, sig):
        fig, ax = _aplot.onsets(sig, self.odx)
        return fig, ax

    def __repr__(self):
        return ''.join('{}: {}\n'.format(mem, self.__getattribute__(mem)) for mem in OnsetResult.__slots__)

    def __str__(self):
        return '<OnsetDetectionResult>'
