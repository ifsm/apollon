"""
Onset detection algorithms
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel
import numpy as np
import pandas as pd
import scipy.signal as _sps

from .. io import io
from .. peak_picking import FilterPeakPicker
from .. signal import features
from .. signal import tools as _ast
from .. signal.spectral import Stft
from .. import fractal as _fractal
from .. import segment as aseg
from .. types import IntArray, FloatArray, PathType
from . import models
from .. models import PeakPickingParams


class OnsetDetector(ABC):
    """Base class for onset detectors"""

    def __init__(self) -> None:
        self._odf: pd.DataFrame
        self._peaks: IntArray
        self._ppkr: FilterPeakPicker
        self._params: BaseModel

    @property
    def odf(self) -> pd.DataFrame:
        """Return onset detection function"""
        return self._odf

    @property
    def onsets(self) -> pd.DataFrame:
        """Return index of each detected onset

        The resulting DataFrame has two columns:
        'frame' is number of the center frame of the segment in which
        the onset was detected.

        'time' is the time difference between the center frame of the segment
        in which the onset was detected and the start of the audio signal.

        The data frame index represents the segments.

        Returns:
            Onset frame index and time
        """
        return self._odf.iloc[self._peaks][['frame', 'time']]

    @property
    def params(self) -> BaseModel:
        """Return initial parameters"""
        return self._params

    @abstractmethod
    def _compute_odf(self, inp: FloatArray) -> pd.DataFrame:
        return NotImplemented

    def detect(self, inp: FloatArray) -> None:
        """Detect onsets."""
        self._odf = self._compute_odf(inp)
        self._peaks = self._ppkr.detect(self._odf['value'].to_numpy().squeeze()).astype(np.int64)

    def to_csv(self, path: PathType) -> None:
        """Serialize odf in csv format.

        Args:
            path: Path to save location
        """
        self.odf.to_csv(path)

    def to_json(self, path: PathType) -> None:
        """Serialize odf in JSON format

        Args:
            path: Path to save location
        """
        self.odf.to_json(path)

    def to_pickle(self, path: PathType) -> None:
        """Serialize object to pickle file

        Args:
            path: Path to save location
        """
        io.save_to_pickle(self, path)

    def plot(self, mode: str ='time') -> None:
        """Plot ODF against time or index

        Args:
            mode:  Either `time`, or `index`
        """
        raise NotImplementedError


class EntropyOnsetDetector(OnsetDetector):
    """Detect onsets based on entropy maxima.
    """
    def __init__(self, fps: int, m_dims: int = 3, delay: int = 10,
                 bins: int = 10, n_perseg: int = 1024, n_overlap: int = 512,
                 pp_params: PeakPickingParams | None = None) -> None:
        """Detect onsets as local maxima of information entropy of consecutive
        windows.

        Be sure to set ``n_perseg`` and ``hop_size`` according to the
        sampling rate of the input signal.

        Params:
            fps:         Sample rate
            m_dim:       Embedding dimension
            bins:        Boxes per axis
            delay:       Embedding delay
            n_perseg:    Length of segments in samples
            hop_size:    Displacement in samples
            smooth:      Smoothing filter length
        """
        super().__init__()
        self.fps = fps
        self.m_dims = m_dims
        self.bins = bins
        self.delay = delay
        self.cutter = aseg.Segmentation(n_perseg, n_overlap)

        self._params = models.EntropyODParams(fps=fps, m_dim=m_dims,
                                       delay=delay, bins=bins,
                                       n_perseg=n_perseg, n_overlap=n_overlap)

        if pp_params:
            self._ppkr = FilterPeakPicker(**pp_params.dict())
        else:
            self._ppkr = FilterPeakPicker()

    def _compute_odf(self, inp: FloatArray) -> pd.DataFrame:
        """Compute onset detection function as the information entropy of
        ``m_dims``-dimensional delay embedding per segment.

        Args:
            inp:  Audio data

        Returns:
            Onset detection function
        """
        segs = self.cutter.transform(inp)
        odf = np.empty((segs.n_segs, 3))
        for i, seg in enumerate(segs):
            emb = _fractal.delay_embedding(seg.squeeze(), self.delay, self.m_dims)
            odf[i, 0] = segs.center(i)
            odf[i, 1] = odf[i, 0] / self.fps
            odf[i, 2] = _fractal.embedding_entropy(emb, self.bins)
            odf[i, 2] = np.maximum(odf[i, 2], odf[i, 2].mean())
        return pd.DataFrame(data=odf, columns=['frame', 'time', 'value'])


class FluxOnsetDetector(OnsetDetector):
    """Onset detection based on spectral flux """

    def __init__(self, fps: int, window: str = 'hamming', n_perseg: int = 1024,
                 n_overlap: int = 512, pp_params: PeakPickingParams | None = None) -> None:
        """Detect onsets as local maxima in the energy difference of
        consecutive stft time steps.

        Args:
            fps:        Sample rate
            window:     Name of window function
            n_perseg:   Samples per segment
            n_overlap:  Numnber of overlapping samples per segment
            pp_params:  Keyword args for peak picking
        """
        super().__init__()
        self._stft = Stft(fps, window, n_perseg, n_overlap)
        self._params = models.FluxODParams(fps=fps, window=window,
                                           n_perseg=n_perseg,
                                           n_overlap=n_overlap)
        if pp_params:
            self._ppkr = FilterPeakPicker(**pp_params.dict())
        else:
            self._ppkr = FilterPeakPicker()

    def _compute_odf(self, inp: FloatArray) -> pd.DataFrame:
        """Onset detection function based on spectral flux

        Args:
            inp:  Audio data

        Returns:
            Onset detection function
        """
        sxx = self._stft.transform(inp)
        flux = features.spectral_flux(sxx.abs, total=True)
        times = sxx.times.squeeze()
        odf = {'frame': (times * sxx.params.fps).astype(int),
               'time': times,
               'value': np.maximum(flux.squeeze(), flux.mean())}
        return pd.DataFrame(odf)
