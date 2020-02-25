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
from typing import Union

import matplotlib.pyplot as _plt
import numpy as _np
import scipy.signal as _sps

from .. segment import Segmentation
from .. types import Array as _Array
from . import tools as _sigtools
from . container import STParams


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
    win = _np.expand_dims(_sps.get_window(window, n_sig), 1)
    bins = _np.fft.rfft(sig*win, n_fft, axis=0)

    if norm:
        bins = bins / _np.sqrt(_np.square(win.sum())) * 2

    return bins


class Spectrum:
    """FFT Spectrum."""
    def __init__(self, params: STParams, bins: _np.ndarray) -> None:
        """Create a new spectrum.

        Args:
            bins:    FFT bins.
            params:  Configuration object.
        """
        if not isinstance(params, STParams):
            raise TypeError('Expected type STParams')
        if not isinstance(bins, _np.ndarray):
            raise TypeError('Expected numpy array.')
        self._params = params
        self._bins = bins

    @property
    def abs(self) -> _Array:
        """Compute magnitude spectrum."""
        return self.__abs__()

    @property
    def bins(self) -> _Array:
        """Raw FFT bins."""
        return self._bins

    @property
    def d_frq(self) -> Union[float, None]:
        """Retrun the frequency resolution."""
        if self._params.n_fft is None:
            return None
        return self._params.fps / self._params.n_fft

    @property
    def frqs(self) -> _Array:
        """Frequency axis."""
        return _np.fft.rfftfreq(self._params.n_fft,
                                1.0/self._params.fps).reshape(-1, 1)

    @property
    def params(self) -> STParams:
        """Initial parameters."""
        return self._params

    @property
    def phase(self):
        """Compute phase spectrum."""
        if self._bins is None:
            return None
        return _np.angle(self._bins)

    @property
    def power(self):
        """Compute power spectrum."""
        return _np.square(self.__abs__())

    @property
    def centroid(self):
        """Compute spectral centroid."""
        return _np.multiply(self.abs, self.frqs).sum() / self.abs.sum()

    def plot(self, fmt='-'):
        """Plot the spectrum."""
        _plt.plot(self.frqs, self.abs, fmt)

    def __abs__(self):
        return _np.absolute(self._bins)

    def __getitem__(self, key):
        return self._bins[key]

    def __len__(self):
        return self._bins.shape[0]

    def __repr__(self):
        return 'Spectrum()'


class Spectrogram(Spectrum):
    def __init__(self, params: STParams, bins: _np.ndarray) -> None:
        super().__init__(params, bins)

    @property
    def n_segments(self) -> int:
        return self._bins.shape[1]

    @property
    def times(self):
        """Compute time axis.
        Features are mapped to the center of each segment
        """

        if self._params.extend:
            start = 0
            stop = self._params.n_overlap * self.n_segments
        else:
            start = self._params.n_perseg / 2
            stop = self._params.n_overlap * self.n_segments + self._params.n_overlap
        step = self._params.n_perseg - self._params.n_overlap
        frame_time = _np.arange(start, stop, step)
        return _np.expand_dims(frame_time / float(self._params.fps), 0)


class _FastFourierTransform:
    """Discrete Fourier Transform"""
    def __init__(self, params: STParams) -> None:
        """Create a new spectrum.

        Args:
            params:  Configuration object.
        """
        if not isinstance(params, STParams):
            raise TypeError('Expected type STParams')
        self._params = params

    def transform(self, inp: _Array) -> _np.ndarray:
        """Transform the input array.

        Args:
            inp:  Two dimensional input array. FFT is performed along the rows.
        """
        inp = _np.atleast_2d(inp)
        if inp.ndim > 2:
            raise ValueError(f'Input array has {inp.dim} dimensions, but it '
                             'should have two at max.')

        if self._params.n_fft is None:
            self._params.n_fft = inp.shape[0]

        return fft(inp, self._params.window, self._params.n_fft)


class Dft(_FastFourierTransform):
    """Discrete Fourier Transform."""
    def __init__(self, fps: int, window: str = 'hamming',
                 n_fft: int = None) -> None:
        """Create a new spectrum.

        Args:
            params:  Initial parameters
        """
        params = STParams(fps, window, n_fft=n_fft)
        super().__init__(params)

    def transform(self, data: _np.ndarray) -> Spectrum:
        return Spectrum(self._params, super().transform(data))


class Stft(_FastFourierTransform):
    """Short Time Fourier Transform of AudioFile."""
    def __init__(self, params: STParams) -> None:
        """Create a new spectrogram.

        Args:
            params:  Initial parameters
        """
        super().__init__(params)
        self._seg = Segmentation(params.n_perseg, params.n_overlap,
                                 params.extend, params.pad)

    def transform(self, data: _np.ndarray) -> Spectrogram:
        segs = self._seg.transform(data)
        return Spectrogram(self._params, super().transform(segs._segs))


class StftSegments(_FastFourierTransform):
    """Short Time Fourier Transform on already segmented audio."""
    def __init__(self, params: STParams) -> None:
        super().__init__(params)

    def transform(self, segments) -> Spectrogram:
        self._params.extend = segments._params.extend
        self._params.pad = segments._params.pad
        return Spectrogram(self._params, super().transform(segments._segs))


