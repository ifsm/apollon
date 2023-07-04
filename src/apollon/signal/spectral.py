"""
Generic Discrete Fourier Transforms for real input
"""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as _plt
import numpy as np
import scipy.signal as _sps

from . models import DftParams, StftParams, SpectralTransformParams
from .. segment import Segmentation, Segments
from .. types import FloatArray, IntArray, ComplexArray
from .. signal import features
from .. models import SegmentationParams


def fft(sig: FloatArray, window: str | None = None, n_fft: int | None = None,
        norm: bool = True) -> ComplexArray:
    """Compute the Discrete Fouier Transform for real input

    This is a simple wrapper around ``numpy.fft.rfft``. Input signal must
    be two-dimensional. FTT is performed along the rows.

    Args:
        sig:     Two-dimensional input array
        n_fft:   FFT length in samples
        window:  Name of window function
        norm:    If True, scale such that a sinusodial signal with unit
                 aplitude has unit amplitude in the spectrum

    Returns:
        FFT bins

    Raises:
        AttributeError
    """
    if sig.ndim != 2:
        raise ValueError(f'Input array has {sig.ndim} dimensions. However,'
                         ' ``fft`` expects two-dimensional array.')
    n_sig = sig.shape[0]
    if n_fft is None:
        n_fft = n_sig

    if window is None:
        window = 'rect'

    win = np.expand_dims(_sps.get_window(window, n_sig), 1)
    bins = np.fft.rfft(sig*win, n_fft, axis=0)

    if norm:
        bins = bins / np.sqrt(np.square(win.sum())) * 2

    return bins


class TransformResult(ABC):
    """Base class for transformation results"""
    def __init__(self, bins: ComplexArray) -> None:
        self._bins = bins
        self._params: SpectralTransformParams
        self._inp_size: int

    @property
    def abs(self) -> FloatArray:
        """Compute magnitude spectrum"""
        return abs(self)

    @property
    def bins(self) -> ComplexArray:
        """Raw FFT bins"""
        return self._bins

    @property
    def d_frq(self) -> float:
        """Retrun the frequency resolution"""
        return int(self._params.fps) / self._n_fft

    @property
    def frqs(self) -> FloatArray:
        """Frequency axis"""
        return np.fft.rfftfreq(self._n_fft,
                               1.0/self._params.fps).reshape(-1, 1)

    @property
    @abstractmethod
    def params(self) -> SpectralTransformParams:
        """Initial parameters"""
        return self._params

    @property
    def phase(self) -> FloatArray:
        """Compute phase spectrum"""
        if self._bins is None:
            return None
        return np.angle(self._bins)

    @property
    def power(self) -> FloatArray:
        """Compute power spectrum"""
        return np.square(self.abs)

    @property
    def centroid(self) -> FloatArray:
        """Compute spectral centroid"""
        return features.spectral_centroid(self.frqs, self.abs)

    @property
    def _n_fft(self) -> int:
        """Compute the FFT length considering ``n_fft`` was ``None``."""
        if self._params.n_fft is None:
            n_fft = self._inp_size
        else:
            n_fft = self._params.n_fft
        return n_fft

    def __abs__(self) -> FloatArray:
        return np.absolute(self._bins)

    def __getitem__(self, key: int) -> ComplexArray:
        return np.asarray(self._bins[key]).astype(np.complex_)

    def __len__(self) -> int:
        return self._bins.shape[0]


class Spectrum(TransformResult):
    """FFT Spectrum"""
    def __init__(self, params: DftParams, bins: ComplexArray,
                 inp_size: int) -> None:
        """Representation of DFT bins with frequency axis

        Args:
            bins:      DFT bins
            params:    DFT parameters
            inp_size:  Length of original signal
        """
        super().__init__(bins)
        if not isinstance(params, DftParams):
            raise TypeError('Expected type ``DftParams``')
        if not isinstance(bins, np.ndarray):
            raise TypeError('Expected numpy array')
        self._params: DftParams = params
        self._inp_size = inp_size

    @property
    def params(self) -> DftParams:
        return self._params

    def plot(self, fmt: str = '-') -> None:
        """Plot the spectrum"""
        _plt.plot(self.frqs, self.abs, fmt)

    def __repr__(self) -> str:
        return f'Spectrum({self._params})'


class Spectrogram(TransformResult):
    """Result of Short Time Fourier Transform"""
    def __init__(self, params: StftParams, bins: ComplexArray,
                 inp_size: int) -> None:
        """Representation of DFT bins with time and frequency axis

        Args:
            params:    Set of params
            bins:      FFT bins
            inp_size:  Length time domain signal
        """
        super().__init__(bins)
        self._params: StftParams = params
        self._inp_size = inp_size

    @property
    def n_segments(self) -> int:
        """Return number of segments"""
        return self._bins.shape[1]

    @property
    def index(self) -> IntArray:
        """Center index regarding original signal per bin"""
        if self._params.extend:
            offset = 0
        else:
            offset = self._params.n_perseg // 2
        return (offset + np.arange(self._bins.shape[1]) *
                (self._params.n_perseg - self._params.n_overlap))

    @property
    def times(self) -> FloatArray:
        """Time axis"""
        return self.index / self._params.fps

    @property
    def params(self) -> StftParams:
        return self._params

    def __repr__(self) -> str:
        return f'Spectrogram({self._params})'


class SpectralTransform(ABC):
    """Base class for spectral transforms"""
    def __init__(self) -> None:
        """SpectralTransform base class

        Args:
            params:  Parameter object
        """
        self._params: SpectralTransformParams

    @abstractmethod
    def transform(self, data: Any) -> TransformResult:
        """Transform ``data`` to spectral domain"""

    @property
    @abstractmethod
    def params(self) -> SpectralTransformParams:
        """Return parameters"""
        return self._params


class Dft(SpectralTransform):
    """Discrete Fourier Transform"""
    def __init__(self, fps: int, window: str | None = None,
                 n_fft: int | None = None, norm: bool = True) -> None:
        """Create a new spectrum

        Args:
            fps:     Sample rate
            window:  Name of window function
            n_fft:   FFT length
            norm:    If ``True``, normalize the spectrum
        """
        super().__init__()
        self._params: DftParams = DftParams(fps=fps, window=window, n_fft=n_fft, norm=norm)

    def transform(self, data: FloatArray) -> Spectrum:
        """Transform ``data`` to spectral domain."""
        bins = fft(data, self.params.window, self.params.n_fft, norm=self.params.norm)
        return Spectrum(self.params, bins, data.shape[0])

    @property
    def params(self) -> DftParams:
        return self._params


class Stft(SpectralTransform):
    """Short Time Fourier Transform of AudioFile."""
    def __init__(self, fps: int, n_perseg: int, n_overlap: int,
                 window: str | None = None,
                 n_fft: int | None = None, extend: bool = True,
                 pad: bool = True) -> None:
        # pylint: disable = R0913
        """Create a new spectrogram.

        Args:
            params:  Initial parameters
        """
        super().__init__()
        self._params: StftParams = StftParams(fps=fps, window=window, n_fft=n_fft,
                                    n_perseg=n_perseg, n_overlap=n_overlap,
                                    extend=extend, pad=pad)
        self._cutter = Segmentation(self.params.n_perseg, self.params.n_overlap,
                                    self.params.extend, self.params.pad)

    def transform(self, data: FloatArray) -> Spectrogram:
        """Transform ``data`` to spectral domain"""
        segs = self._cutter.transform(data)
        bins = fft(segs.data, self.params.window, self.params.n_fft)
        return Spectrogram(self._params, bins, segs.params.n_perseg)

    @property
    def params(self) -> StftParams:
        return self._params


class StftSegments(SpectralTransform):
    """Short Time Fourier Transform on already segmented audio"""
    def __init__(self, fps: int, seg_params: SegmentationParams, window: str | None = None,
                 n_fft: int | None = None) -> None:
        """Create a new ``Spectrogram`` from ``Segments``

        Args:
            fps:     Sample rate
            window:  Name of window function
            n_fft:   FFT length
        """
        super().__init__()
        self._params: StftParams = StftParams(fps=fps, window=window,
                                              n_fft=n_fft, **seg_params.dict())

    def transform(self, segments: Segments) -> Spectrogram:
        """Transform ``data`` to spectral domain"""
        bins = fft(segments.data, self._params.window, self._params.n_fft)
        return Spectrogram(self._params, bins, segments._params.n_perseg)

    @property
    def params(self) -> StftParams:
        return self._params
