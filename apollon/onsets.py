#!/usr/bin/python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as _plt
import numpy as _np
import scipy.signal as _sps

from apollon import aplot as _aplot
from apollon import fractal as _fractal
from apollon import segment as _segment


def detect(sig, tau=10, m=3, bins=50, wlen=16, whs=8, order=22):
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

    Return:
        (tuple) array of indices, list of entropy values
    """
    # segment audio
    chunks = _segment.by_ms_with_hop(sig, wlen, whs)

    # calculate entropy for each chunk
    H = _np.empty(len(chunks))
    for i, ch in enumerate(chunks):
        em = _fractal.embedding(ch, tau, m=m, mode='wrap')
        H[i] = _fractal.pps_entropy(em, bins)

    # Take imaginary part of the Hilbert transform of the enropy
    H = _sps.hilbert(H).imag

    # pick the peaks
    w = _np.absolute(H)    # use absolute val to consider negative peaks, too
    odf = _sps.argrelmax(w, order=order)[0]

    # calculate onset position to be in the middle of chunks
    onsets_idx = [(i+j)//2 for (i, j) in chunks.get_limits()[odf]]

    return onsets_idx, H
