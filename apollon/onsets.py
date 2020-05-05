"""
apollon/onsets.py -- Onset detection routines.
Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net

Classes:
    OnsetDetector           Base class for onset detection.
    EntropyOnsetDetector    Onset detection based on phase pace entropy estimation.
    FluxOnsetDetector       Onset detection based on spectral flux.

Functions:
    peak_picking            Identify local peaks in time series.
    evaluate_onsets         Evaluation of onset detection results given ground truth.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
import scipy.signal as _sps

from . container import Params
from . signal import features
from . import fractal as _fractal
from . import segment as _segment
from . signal import tools as _ast
from . signal.spectral import Stft, StftParams
from . types import Array


T = TypeVar('T')


@dataclass
class PeakPickingParams(Params):
    n_before: int
    n_after: int
    alpha: float
    delta: float


@dataclass
class FluxOnsetDetectorParams(Params):
    stft_params: StftParams
    pp_params: PeakPickingParams



pp_params = {'n_before': 10, 'n_after': 10, 'alpha': .1,
                  'delta': .1}

class OnsetDetector:
    """Onset detection base class.
    """
    def __init__(self):
        pass

    def detect(self, inp: Array) -> None:
        """Detect onsets."""
        self._odf = self._compute_odf(inp)
        self._peaks = peak_picking(self._odf.to_numpy().squeeze())


class EntropyOnsetDetector(OnsetDetector):
    """Detect onsets based on entropy maxima.
    """
    def __init__(self, inp: Array, m_dims: int = 3, delay: int = None,
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

    def _odf(self, inp: Array) -> Array:
        """Compute onset detection function as the information entropy of
        ``m_dims``-dimensional delay embedding per segment.

        Args:
            inp:    Audio data.

        Returns:
            Onset detection function.
        """
        segments = _segment.by_samples(inp, self.n_perseg, self.hop_size)
        odf = np.empty(segments.shape[0])
        for i, seg in enumerate(segments):
            emb = _fractal.delay_embedding(seg, self.delay, self.m_dims)
            odf[i] = _fractal.embedding_entropy(emb, self.bins)
        return np.maximum(odf, odf.mean())


class FluxOnsetDetector(OnsetDetector):
    """Onset detection based on spectral flux.
    """
    def __init__(self, fps: int, window: str = 'hamming', n_perseg: int = 1024,
                 n_overlap: int = 512, pp_params = None) -> None:
        """Detect onsets as local maxima in the energy difference of
        consecutive stft time steps.

        Args:
            fps:          Sample rate.
            stft_params:  Keyword args for STFT.
            pp_params:    Keyword args for peak picking.
        """
        super().__init__()
        self._stft = Stft(fps, window, n_perseg, n_overlap)

    @property
    def index(self) -> Array:
        return self.times * self._stft.params.fps

    @property
    def odf(self) -> Array:
        return self._odf

    @property
    def params(self):
        return self._params

    @property
    def times(self) -> Array:
        return self._odf.iloc[self._peaks].index.to_numpy()

    def _compute_odf(self, inp: Array) -> Array:
        """Onset detection function based on spectral flux.

        Args:
            inp:  Audio data.

        Returns:
            Onset detection function.
        """
        sxx = self._stft.transform(inp)
        flux = features.spectral_flux(sxx.abs, total=True)
        odf = np.maximum(flux.squeeze(), flux.mean())
        return pd.DataFrame(odf, index=sxx.times.squeeze(), columns=['odf'])


class FilterPeakPicker:
    def __init__(self, odf: Array, n_after: int = 10, n_before: int = 10,
                 alpha: float = .1, delta: float=.1) -> Array:
        pass

    def detect(self)
        """Pick local maxima from a numerical time series.

        Pick local maxima from the onset detection function `odf`, which is assumed
        to be an one-dimensional array. Typically, `odf` is the Spectral Flux per
        time step.

        Args:
            odf:         Onset detection function, e.g., Spectral Flux.
            n_after: Window lenght to consider after now.
            n_before:  Window lenght to consider before now.
            alpha:       Smoothing factor. Must be in ]0, 1[.
            delta:       Difference to the mean.

        Return:
            Peak indices.
        """
        g = [0]
        out = []

        for n, val in enumerate(odf):
            # set local window
            idx = np.arange(n-n_before, n+n_after+1, 1)
            window = np.take(odf, idx, mode='clip')

            cond1 = np.all(val >= window)
            cond2 = val >= (np.mean(window) + delta)

            foo = max(val, alpha*g[n] + (1-alpha)*val)
            g.append(foo)
            cond3 = val >= foo

            if cond1 and cond2 and cond3:
                out.append(n)

        return np.array(out)


def evaluate_onsets(targets: Dict[str, np.ndarray],
                    estimates: Dict[str, np.ndarray]
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

    return np.array(out)
