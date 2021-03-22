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
from . io import io
from . signal import features
from . signal import tools as _ast
from . signal.spectral import Stft, StftParams
from . import fractal as _fractal
from . import segment as aseg
from . types import Array, PathType


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
    def __init__(self) -> None:
        self._odf: Optional[pd.DataFrame] = None

    @property
    def odf(self) -> pd.DataFrame:
        return self._odf

    @property
    def onsets(self) -> pd.DataFrame:
        """Returns the index of each detected onset.

        The resulting data frame has two columns:
        `frame` is number of the center frame of the segment in which
        the onset was detected.

        `time` is the time difference between the center frame of the segment
        in which the onset was detected and the start of the audio signal.

        The data frame index represents the segments.

        Returns:
            Index of each onset as data frame.
        """
        return self._odf.iloc[self._peaks][['frame', 'time']]

    @property
    def params(self):
        """Return initial parameters."""
        return self._params

    def detect(self, inp: Array) -> None:
        """Detect onsets."""
        self._odf = self._compute_odf(inp)
        self._peaks = self._ppkr.detect(self._odf['value'].to_numpy().squeeze())

    def to_csv(self, path: PathType) -> None:
        """Serialize odf in csv format.

        Args:
            path: Path to save location.
        """
        self.odf.to_csv(path)

    def to_json(self, path: PathType) -> None:
        """Serialize odf in JSON format.

        Args:
            path: Path to save location.
        """
        self.odf.to_json(path)

    def to_pickle(self, path: PathType) -> None:
        """Serialize object to pickle file.

        Args:
            path: Path to save location.
        """
        io.save_to_pickle(self, path)

    def plot(self, mode: str ='time') -> None:
        """Plot odf against time or index.

        Args:
            mode:  Either `time`, or `index`.
        """
        raise NotImplementedError


class EntropyOnsetDetector(OnsetDetector):
    """Detect onsets based on entropy maxima.
    """
    def __init__(self, fps: int, m_dims: int = 3, delay: int = 10,
                 bins: int = 10, n_perseg: int = 1024, n_overlap: int = 512,
                 pp_params: Optional[dict] = None) -> None:
        """Detect onsets as local maxima of information entropy of consecutive
        windows.

        Be sure to set ``n_perseg`` and ``hop_size`` according to the
        sampling rate of the input signal.

        Params:
            fps:         Audio signal.
            m_dim:       Embedding dimension.
            bins:        Boxes per axis.
            delay:       Embedding delay.
            n_perseg:    Length of segments in samples.
            hop_size:    Displacement in samples.
            smooth:      Smoothing filter length.
        """
        super().__init__()
        self.fps = fps
        self.m_dims = m_dims
        self.bins = bins
        self.delay = delay
        self.cutter = aseg.Segmentation(n_perseg, n_overlap)

        if pp_params:
            self._ppkr = FilterPeakPicker(**pp_params)
        else:
            self._ppkr = FilterPeakPicker()

    def _compute_odf(self, inp: Array) -> Array:
        """Compute onset detection function as the information entropy of
        ``m_dims``-dimensional delay embedding per segment.

        Args:
            inp:  Audio data.

        Returns:
            Onset detection function.
        """
        segs = self.cutter.transform(inp)
        odf = np.empty((segs.n_segs, 3))
        for i, seg in enumerate(segs):
            emb = _fractal.delay_embedding(seg.squeeze(), self.delay, self.m_dims)
            odf[i, 0] = segs.center(i)
            odf[i, 0] = odf[i, 0] / self.fps
            odf[i, 2] = _fractal.embedding_entropy(emb, self.bins)
        odf[i, 2] = np.maximum(odf[i, 2], odf[i, 2].mean())
        return pd.DataFrame(data=odf, columns=['frame', 'time', 'value'])


class FluxOnsetDetector(OnsetDetector):
    """Onset detection based on spectral flux.
    """
    def __init__(self, fps: int, window: str = 'hamming', n_perseg: int = 1024,
                 n_overlap: int = 512, pp_params: Optional[dict] = None) -> None:
        """Detect onsets as local maxima in the energy difference of
        consecutive stft time steps.

        Args:
            fps:        Sample rate.
            window:     Name of window function.
            n_perseg:   Samples per segment.
            n_overlap:  Numnber of overlapping samples per segment.
            pp_params:  Keyword args for peak picking.
        """
        super().__init__()
        self._stft = Stft(fps, window, n_perseg, n_overlap)
        if pp_params:
            self._ppkr = FilterPeakPicker(**pp_params)
        else:
            self._ppkr = FilterPeakPicker()

    def _compute_odf(self, inp: Array) -> Array:
        """Onset detection function based on spectral flux.

        Args:
            inp:  Audio data.

        Returns:
            Onset detection function.
        """
        sxx = self._stft.transform(inp)
        flux = features.spectral_flux(sxx.abs, total=True)
        times = sxx.times.squeeze()
        odf = {'frame': (times * sxx.params.fps).astype(int),
               'time': times,
               'value': np.maximum(flux.squeeze(), flux.mean())}
        return pd.DataFrame(odf)


class FilterPeakPicker:
    def __init__(self, n_after: int = 10, n_before: int = 10,
                 alpha: float = .1, delta: float=.1) -> None:
        self.n_after = n_after
        self.n_before = n_before
        self.alpha = alpha
        self.delta = delta

    def detect(self, inp: Array) -> Array:
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

        for n, val in enumerate(inp):
            # set local window
            idx = np.arange(n-self.n_before, n+self.n_after+1, 1)
            window = np.take(inp, idx, mode='clip')

            cond1 = np.all(val >= window)
            cond2 = val >= (np.mean(window) + self.delta)

            foo = max(val, self.alpha*g[n] + (1-self.alpha)*val)
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
