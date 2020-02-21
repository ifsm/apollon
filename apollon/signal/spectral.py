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

from .. types import Array as _Array
from . import tools as _sigtools
from . import container


def fft(sig, window: str = None, n_fft: int = None,
        norm: bool = True):
    """Compute the Discrete Fourier Transform for real input.

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
    def __init__(self, params: container.SpectrumParams) -> None:
        """Create a new spectrum.

        Args:
            params:  Configuration object.
        """
        if not isinstance(params, container.SpectrumParams):
            raise TypeError('Expected type SpectrumParams')
        self._params = params
        self._frqs = None
        self._bins = None

    def transform(self, inp: _Array) -> None:
        """Transform the input array.

        Args:
            inp:  Two dimensional input array. FFT is performed along the rows.
        """
        inp = _np.atleast_2d(inp)
        if inp.ndim > 2:
            raise ValueError(f'Input array has {inp.dim} dimensions, but it '
                             'should have two at max.')

        if self._params.n_fft is None:
            size = inp.shape[0]
        else:
            size = self._params.n_fft

        self._bins = fft(inp, self._params.window, self._params.n_fft)
        self._frqs = _np.fft.rfftfreq(size, 1.0/self._params.fps).reshape(-1, 1)

        trim = _sigtools.trim_range(self.d_frq, self.params.lcf, self.params.ucf)
        self._bins = self._bins[trim]
        self._frqs = self._frqs[trim]

    @property
    def abs(self) -> _Array:
        """Compute magintude spectrum."""
        return self.__abs__()

    @property
    def bins(self) -> _Array:
        """Raw FFT bins."""
        return self._bins

    @property
    def d_frq(self) -> Union[int, None]:
        """Retrun the frequency resolution."""
        try:
            return self._frqs[1, 0] - self._frqs[0, 0]
        except TypeError:
            return None

    @property
    def frqs(self) -> _Array:
        """Frequency axis."""
        return self._frqs

    @property
    def params(self) -> container.SpectrumParams:
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

    def _trim(self, lcf: float = None, ucf: float = None) -> None:
        try:
            lcf = int(lcf//self.d_frq)
        except TypeError:
            lcf = None

        try:
            ucf = int(ucf//self.d_frq)
        except TypeError:
            ucf = None

        trim_range = slice(lcf, ucf)
        self._frqs = self._frqs[trim_range]
        self._bins = self._bins[trim_range]

    def __abs__(self):
        if self._bins is None:
            return None
        return _sigtools.limit(_np.absolute(self._bins), self._params.ldb,
                               self._params.udb)

    def __getitem__(self, key):
        return self._bins[key]

    def __len__(self):
        return self._bins.shape[0]

    def __repr__(self):
        return 'Spectrum()'
