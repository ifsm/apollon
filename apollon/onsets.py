from typing import Dict, Tuple

import numpy as _np
import scipy.signal as _sps

from . import fractal as _fractal
from . import segment as _segment
from . signal.spectral import stft as _stft
from . signal.tools import trim_spectrogram as _trim_spectrogram
from . types import Array as _Array


class OnsetDetector:
    def __init__(self):
        pass

    def _compute_odf(self):
        pass

    def _detect(self):
        """Detect local maxima of the onset detection function.

        Returns:
            Position of onset as index of the odf.
        """
        return peak_picking(self.odf)

    def index(self) -> _Array:
        """Compute onset index.

        Onset values are centered within the detection window.

        Returns:
            Onset position in samples
        """
        return self.peaks * self.hop_size + self.n_perseg // 2

    def times(self, fps: int) -> _Array:
        """Compute time code im ms for each onset give the sample rate.

        Args:
            fps: Sample rate.

        Returns:
            Time code of onsets.
        """
        return self.index() / fps


class EntropyOnsetDetector(OnsetDetector):

    def __init__(self, inp: _Array, delay: int = 10, m_dims: int = 3, bins: int = 10,
                 n_perseg: int = 1024, hop_size: int = 512) -> None:
        """Detect onsets based on entropy maxima.

        Args:
            inp:        Audio signal.
            delay:      Embedding delay.
            m_dim:      Embedding dimension.
            bins:       Boxes per axis.
            n_perseg:   Length of segments in samples.
            hop_size:   Displacement in samples.
            smooth:     Smoothing filter length.
        """
        self.delay = delay
        self.m_dims = m_dims
        self.bins = bins
        self.n_perseg = n_perseg
        self.hop_size = hop_size
        self.smooth = smooth

        self.odf = self._odf(inp)
        self.peaks = self._detect()


    def _odf(self, inp: _Array) -> _Array:
        """Compute onset detection function as the information entropy of ```m_dims```-dimensional
        delay embedding per segment.

        Args:
            inp:    Audio data.

        Returns:
            Onset detection function.
        """
        segments = _segment.by_samples(inp, self.n_perseg, self.hop_size)
        odf = _np.empty(segments.shape[0])
        for i, seg in enumerate(segments):
            emb = _fractal.embedding(seg, self.delay, self.m_dims, mode='wrap')
            odf[i] = _fractal.embedding_entropy(emb, self.bins)
        return _np.maximum(odf, odf.mean())


class FluxOnsetDetector(OnsetDetector):
    """Onset detection based on spectral flux."""

    def __init__(self, inp: _Array, fps: int, window: str = 'hamming', n_perseg: int = 2048,
                 hop_size: int = 441):

        self.fps = fps
        self.window = window
        self.n_perseg = n_perseg
        self.hop_size = hop_size
        self.smooth = smooth

        self.odf = self._odf(inp)
        self.peaks = self._detect()


    def _odf(self, inp: _Array) -> _Array:
        """Onset detection function based on spectral flux.

        Args:
            inp:    Audio data.

        Returns:
            Onset detection function.
        """
        spctrgrm = _stft(inp, self.fps, self.window, self.n_perseg, self.hop_size)
        sb_flux, _ = _trim_spectrogram(spctrgrm.flux(subband=True), spctrgrm.frqs, 80, 10000)
        odf = sb_flux.sum(axis=0)
        return _np.maximum(odf, odf.mean())


def peak_picking(odf: _Array, post_window: int = 10, pre_window: int = 10, alpha: float = .1,
                 delta: float=.1) -> _Array:
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
                    estimates: Dict[str, _np.ndarray]) -> Tuple[float, float,
                                                                float]:
    """Evaluate the performance of an onset detection.

    Args:
        targets:    of ground truth onset times, with
                            keys   == file names, and
                            values == target onset times in ms.

        estimates:  of estimated onsets times, with
                            keys   == file names, and
                            values == estimated onset times in ms.

    Return:
        Tupel of precison, recall, f-measure.
    """
    out = []
    for name, tvals in targets.items():
        od_eval = _me.onset.evaluate(tvals, estimates[name])
        out.append([i for i in od_eval.values()])

    return _np.array(out)
