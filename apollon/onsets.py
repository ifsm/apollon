#!/usr/bin/python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as _plt
import numpy as _np
import scipy.signal as _sps

from apollon import fractal as _fractal
from apollon import segment as _segment
from apollon import tools as _tools


class OnsetDetector:

    __slots__ = ['audio_file_name', 'bins', 'idx', 'm', 'odf', 'order', 'tau',
                 'time_stamp', 'window_length', 'window_hop_size']

    def __init__(self):
        pass

    def detect(self, sig, tau=10, m=3, bins=50, wlen=16, whs=8, order=22):
        """Detect note onsets in (percussive) music as local maxima of information
        entropy.

        Params:
            sig      (array-like) Audio signal.
            tau      (int) Phase-space: delay parameter in samples.
            m        (int) Phase-space: number of dimensions.
            bins     (int) Phase-space: number of boxes per axis.
            wlen     (int) Segmentation: window length in ms.
            whs      (int) Segmentation: window displacement in ms.
            order    (int) Peak-picling: Order of filter in samples.
        """
        # meta
        self.audio_file_name = str(sig.file)
        self.bins = bins
        self.m = m
        self.order = order
        self.tau = tau
        self.time_stamp = _tools.time_stamp()
        self.window_hop_size = whs
        self.window_length = wlen


        # segment audio
        chunks = _segment.by_ms_with_hop(sig, self.window_length,
                                         self.window_hop_size)

        # calculate entropy for each chunk
        H = _np.empty(len(chunks))
        for i, ch in enumerate(chunks):
            em = _fractal.embedding(ch, self.tau, m=self.m, mode='wrap')
            H[i] = _fractal.pps_entropy(em, self.bins)

        # Take imaginary part of the Hilbert transform of the enropy
        self.odf = _np.absolute(_sps.hilbert(H).imag)

        # pick the peaks
        peaks, = _sps.argrelmax(self.odf, order=self.order)

        # calculate onset position to be in the middle of chunks
        self.idx = _np.array( [(i+j)//2
                              for (i, j) in chunks.get_limits()[peaks]])
