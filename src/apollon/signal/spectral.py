"""apollon/signal/spectral.py

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net

Provide easy access to frequency spectra obtained by the DFT.

Classes:
    Spectrum
    Spectrogram
    stft

Functions:
    fft:  One-sided Fast fourier transform for real input.
"""
from typing import Any, Union

import matplotlib.pyplot as _plt
import numpy as np
import scipy.signal as _sps

from .. segment import Segmentation, Segments
from .. types import Array, Optional
from . container import Params, DftParams, StftParams


def fft(sig, window: str = None, n_fft: int = None,
        norm: bool = True):
    """Compute the Discrete Fouier Transform for real input.

    This is a simple wrapper around ``numpy.fft.rfft``. Input signal must
    be two-dimensional. FTT is performed along the rows.

    Args:
        sig:     Two-dimensional input array.
        n_fft:   FFT length in samples.
        window:  Name of window function.
        norm:    If True, scale such that a sinusodial signal with unit
                 aplitude has unit amplitude in the spectrum.

    Returns:
        FFT bins.

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


class TransformResult:
    """Base class for transformation results."""
    def __init__(self, params: Any, bins: np.ndarray) -> None:
        self._params = params
        self._bins = bins

    @property
    def abs(self) -> Array:
        """Compute magnitude spectrum."""
        return self.__abs__()

    @property
    def bins(self) -> Array:
        """Raw FFT bins."""
        return self._bins

    @property
    def d_frq(self) -> Union[float, None]:
        """Retrun the frequency resolution."""
        return self._params.fps / self._n_fft

    @property
    def frqs(self) -> Array:
        """Frequency axis."""
        return np.fft.rfftfreq(self._n_fft,
                               1.0/self._params.fps).reshape(-1, 1)

    @property
    def params(self) -> DftParams:
        """Initial parameters."""
        return self._params

    @property
    def phase(self):
        """Compute phase spectrum."""
        if self._bins is None:
            return None
        return np.angle(self._bins)

    @property
    def power(self):
        """Compute power spectrum."""
        return np.square(self.__abs__())

    @property
    def centroid(self):
        """Compute spectral centroid."""
        return np.multiply(self.abs, self.frqs).sum() / self.abs.sum()

    @property
    def _n_fft(self) -> int:
        """Compute the FFT length considering ``n_fft`` was ``None``."""
        if self._params.n_fft is None:
            n_fft = self._inp_size
        else:
            n_fft = self._params.n_fft
        return n_fft

    def __abs__(self) -> Array:
        return np.absolute(self._bins)

    def __getitem__(self, key) -> Array:
        return self._bins[key]

    def __len__(self) -> int:
        return self._bins.shape[0]


class Spectrum(TransformResult):
    """FFT Spectrum."""
    def __init__(self, params: DftParams, bins: np.ndarray,
                 inp_size: int) -> None:
        """Representation of DFT bins with frequency axis.

        Args:
            bins:      DFT bins.
            params:    DFT parameters.
            inp_size:  Length of original signal.
        """
        if not isinstance(params, Params):
            raise TypeError('Expected type ``Params``')
        if not isinstance(bins, np.ndarray):
            raise TypeError('Expected numpy array.')
        super().__init__(params, bins)
        self._inp_size = inp_size

    def plot(self, fmt='-') -> None:
        """Plot the spectrum."""
        _plt.plot(self.frqs, self.abs, fmt)

    def __repr__(self) -> str:
        return f'Spectrum({self._params})'


class Spectrogram(TransformResult):
    """Result of Short Time Fourier Transform."""
    def __init__(self, params: StftParams, bins: np.ndarray,
                 inp_size: int) -> None:
        """Representation of DFT bins with time and frequency axis.

        Args:
            params:    Set of params.
            bins:      FFT bins
            inp_size:  Length time domain signal.
        """
        super().__init__(params, bins)
        self._inp_size = inp_size

    @property
    def n_segments(self) -> int:
        """Return number of segments."""
        return self._bins.shape[1]

    @property
    def index(self) -> Array:
        """Center index regarding original signal per bin."""
        if self._params.extend:
            offset = 0
        else:
            offset = self._params.n_perseg // 2
        return (offset + np.arange(self._bins.shape[1]) *
                (self._params.n_perseg - self._params.n_overlap))

    @property
    def times(self) -> Array:
        """Time axis."""
        return self.index / self._params.fps

    def __repr__(self) -> str:
        return f'Spectrogram({self._params})'


class SpectralTransform:
    """Base class for spectral transforms."""
    def __init__(self, params: Params):
        """SpectralTransform base class.

        Args:
            params:  Parameter object.
        """
        self._params = params

    def transform(self, data: np.ndarray):
        """Transform ``data`` to spectral domain."""

    @property
    def params(self) -> Params:
        """Return parameters."""
        return self._params


class Dft(SpectralTransform):
    """Discrete Fourier Transform."""
    def __init__(self, fps: int, window: str,
                 n_fft: Optional[int] = None) -> None:
        """Create a new spectrum.

        Args:
            fps:     Sample rate.
            window:  Name of window function.
            n_fft:   FFT length.
        """
        super().__init__(DftParams(fps, window, n_fft))

    def transform(self, data: np.ndarray) -> Spectrum:
        """Transform ``data`` to spectral domain."""
        bins = fft(data, self.params.window, self.params.n_fft)
        return Spectrum(self._params, bins, data.shape[0])


class Stft(SpectralTransform):
    """Short Time Fourier Transform of AudioFile."""
    def __init__(self, fps: int, window: str,
                 n_perseg: int, n_overlap: int,
                 n_fft: Optional[int] = None, extend: bool = True,
                 pad: bool = True) -> None:
        """Create a new spectrogram.

        Args:
            params:  Initial parameters
        """
        super().__init__(StftParams(fps, window, n_fft, n_perseg,
                                    n_overlap, extend, pad))
        self._cutter = Segmentation(self.params.n_perseg, self.params.n_overlap,
                                    self.params.extend, self.params.pad)

    def transform(self, data: np.ndarray) -> Spectrogram:
        """Transform ``data`` to spectral domain."""
        segs = self._cutter.transform(data)
        bins = fft(segs.data, self.params.window, self.params.n_fft)
        return Spectrogram(self._params, bins, segs.params.n_perseg)


class StftSegments(SpectralTransform):
    """Short Time Fourier Transform on already segmented audio."""
    def __init__(self, fps: int, window: str,
                 n_fft: Optional[int] = None) -> None:
        """Create a new ``Spectrogram`` from ``Segments``.

        Args:
            fps:     Sample rate.
            window:  Name of window function.
            n_fft:   FFT length.
        """
        super().__init__(StftParams(fps, window, n_fft))

    def transform(self, segments: Segments) -> Spectrogram:
        """Transform ``data`` to spectral domain."""
        for key, val in segments.params.to_dict().items():
            setattr(self.params, key, val)
        bins = fft(segments.data, self.params.window, self.params.n_fft)
        return Spectrogram(self.params, bins, segments.params.n_perseg)
