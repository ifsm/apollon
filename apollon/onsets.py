# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
apollon/onsets.py -- Onset detection routines.

Classes:
    OnsetDetector           Base class for onset detection.
    EntropyOnsetDetector    Onset detection based on phase pace entropy estimation.
    FluxOnsetDetector       Onset detection based on spectral flux.

Functions:
    peak_picking            Identify local peaks in time series.
    evaluate_onsets         Evaluation of onset detection results given ground truth.
"""
from typing import Dict, Tuple

import numpy as _np
import scipy.signal as _sps

from . import fractal as _fractal
from . import segment as _segment
from . signal import tools as _ast
from . signal.spectral import stft as _stft
from . types import Array as _Array


class OnsetDetector:
    """Onset detection base class.

    Subclasses have to implement an __init__ method to take in custom
    arguments. It necessarily has to call the base classes __init__ method.
    Additionally, subclasses have to implement a custom onset detection
    function named _odf. This method should return an one-dimensional ndarray.
    """
    def __init__(self):
        self.pp_params = {'pre_window': 10, 'post_window': 10, 'alpha': .1,
                          'delta': .1}
        self.align = 'center'

    def _odf(self, inp: _Array) -> _Array:
        pass

    def _detect(self):
        """Detect local maxima of the onset detection function.

        Returns:
            Position of onset as index of the odf.
        """
        return peak_picking(self.odf, **self.pp_params)

    def index(self) -> _Array:
        """Compute onset index.

        Onset values are centered within the detection window.

        Returns:
            Onset position in samples
        """
        left = self.peaks * self.hop_size

        if self.align == 'left':
            return left

        if self.align == 'center':
            return left + self.n_perseg // 2

        if self.align == 'right':
            return left + self.n_perseg

        raise ValueError('Unknown alignment method `{}`.'.format(pp_params['align']))

    def times(self, fps: int) -> _Array:
        """Compute time code im ms for each onset give the sample rate.

        Args:
            fps: Sample rate.

        Returns:
            Time code of onsets.
        """
        return self.index() / fps


class EntropyOnsetDetector(OnsetDetector):
    """Detect onsets based on entropy maxima.
    """
    def __init__(self, inp: _Array, m_dims: int = 3, delay: int = None,
                 bins: int = 10, n_perseg: int = 1024, hop_size: int = 512,
                 pp_params = None) -> None:
        """Detect onsets as local maxima of information entropy of consecutive
        windows.

        Be sure to set ``n_perseg`` and ``hop_size`` according to the
        sampling rate of the input signal.

        Params:
            inp:         Audio signal.
            m_dim:       Embedding dimension.
            bins:        Boxes per axis.
            delay:       Embedding delay.
            n_perseg:    Length of segments in samples.
            hop_size:    Displacement in samples.
            smooth:      Smoothing filter length.
        """
        super().__init__()

        self.m_dims = m_dims
        self.bins = bins
        self.delay = delay
        self.n_perseg = n_perseg
        self.hop_size = hop_size

        if pp_params is not None:
            self.pp_params = pp_params
        self.odf = self._odf(inp)
        self.peaks = self._detect()


    def _odf(self, inp: _Array) -> _Array:
        """Compute onset detection function as the information entropy of
        ``m_dims``-dimensional delay embedding per segment.

        Args:
            inp:    Audio data.

        Returns:
            Onset detection function.
        """
        segments = _segment.by_samples(inp, self.n_perseg, self.hop_size)
        odf = _np.empty(segments.shape[0])
        for i, seg in enumerate(segments):
            emb = _fractal.delay_embedding(seg, self.delay, self.m_dims)
            odf[i] = _fractal.embedding_entropy(emb, self.bins)
        return _np.maximum(odf, odf.mean())


class FluxOnsetDetector(OnsetDetector):
    """Onset detection based on spectral flux.
    """
    def __init__(self, inp: _Array, fps: int, window: str = 'hamming',
                 n_perseg: int = 2048, hop_size: int = 441, cutoff=(80, 10000),
                 n_fft: int = None, pp_params = None):
        """Detect onsets as local maxima in the energy difference of
        consecutive stft time steps.

        Params:
            inp:          Input array.
            fps:          Sample rate.
            window:       Window function.
            n_perseg:     Samples per FFT segment.
            hop_size:     FFT window shift in samples.
            cut_off:      Lower and upper cutoff frequency.
            n_fft:        Number sample points per FFT.
            pp_params:    Key word arguments for peak picking.
        """
        super().__init__()

        self.fps = fps
        self.window = window
        self.n_perseg = n_perseg
        self.hop_size = hop_size

        if n_fft is None:
            self.n_fft = n_perseg
        else:
            self.n_fft = n_fft

        self.cutoff = cutoff

        if pp_params is not None:
            self.pp_params = pp_params

        self.odf = self._odf(inp)
        self.peaks = self._detect()


    def _odf(self, inp: _Array) -> _Array:
        """Onset detection function based on spectral flux.

        Args:
            inp:    Audio data.

        Returns:
            Onset detection function.
        """
        spctrgrm = _stft(inp, self.fps, self.window, self.n_perseg,
                         self.hop_size)
        sb_flux, _ = _ast.trim_spectrogram(spctrgrm.flux(subband=True),
                                          spctrgrm.frqs, *self.cutoff)
        odf = sb_flux.sum(axis=0)
        return _np.maximum(odf, odf.mean())

    def params(self) -> dict:
        _params = ('window', 'n_perseg', 'hop_size', 'n_fft', 'pp_params', 'align')
        out = {param: getattr(self, param) for param in _params}
        out['cutoff'] = {'lower': self.cutoff[0], 'upper': self.cutoff[1]}
        return out


def peak_picking(odf: _Array, post_window: int = 10, pre_window: int = 10,
                 alpha: float = .1, delta: float=.1) -> _Array:
    """Pick local maxima from a numerical time series.

    Pick local maxima from the onset detection function `odf`, which is assumed
    to be an one-dimensional array. Typically, `odf` is the Spectral Flux per
    time step.

    Params:
        odf:         Onset detection function, e.g., Spectral Flux.
        post_window: Window lenght to consider after now.
        pre_window:  Window lenght to consider before now.
        alpha:       Smoothing factor. Must be in ]0, 1[.
        delta:       Difference to the mean.

    Return:
        Peak indices.
    """
    g = [0]
    out = []

    for n, val in enumerate(odf):

        # set local window
        idx = _np.arange(n-pre_window, n+post_window+1, 1)
        window = _np.take(odf, idx, mode='clip')

        cond1 = _np.all(val >= window)
        cond2 = val >= (_np.mean(window) + delta)

        foo = max(val, alpha*g[n] + (1-alpha)*val)
        g.append(foo)
        cond3 = val >= foo

        if cond1 and cond2 and cond3:
            out.append(n)

    return _np.array(out)


def evaluate_onsets(targets: Dict[str, _np.ndarray],
                    estimates: Dict[str, _np.ndarray]
                    ) -> Tuple[float, float, float]:
    """Evaluate onset detection performance.

    This function uses the mir_eval package for evaluation.

    Params:
        targets:    Ground truth onset times, with dict keys being file names,
                    and values being target onset time codes in ms.

        estimates:  Estimated onsets times, with dictkeys being file names,
                    and values being the estimated onset time codes in ms.

    Returns:
        Precison, recall, f-measure.
    """
    out = []
    for name, tvals in targets.items():
        od_eval = _me.onset.evaluate(tvals, estimates[name])
        out.append([i for i in od_eval.values()])

    return _np.array(out)
