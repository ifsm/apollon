# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""apollon/signal/spectral.py

Provide easy access to frequency spectra obtained by the DFT.

Classes:
    Spectrum
    Spectrogram
    stft

Functions:
    fft:  One-sided Fast fourier transform for real input.
"""
import matplotlib.pyplot as _plt
import numpy as _np

from . import features as _features
from . import tools as _sigtools
from .. import container
from .. import _defaults
from .. types import Array as _Array


def fft(sig: _Array, window: str = None, n_fft: int = None) -> _Array:
    """Compute the normalized Discrete Fourier Transform for real input.

    This is a simple wrapper around `numpy.fft.rfft`.

    Args:
        sig:     Input signal.
        n_fft:   FFT length in samples.
        window:  Name of window function.

    Returns:
        FFT bins.

    Raises:
        AttributeError
    """
    sig = _np.atleast_2d(sig).astype('float64')
    n_sig = sig.shape[0]

    if n_fft is None:
        n_fft = n_sig

    if window is not None:
        try:
            win_func = getattr(_np, window)
        except AttributeError:
            raise AttributeError(f'Unknown window function `{window}`.')
        sig *= _np.expand_dims(win_func(n_sig), 1)

    bins = _np.fft.rfft(sig, n_fft, axis=0) / float(n_fft)

    if n_fft % 2:
        bins *= 2.0
    else:
        bins[:-1] *= 2.0
    return bins


class Spectrum:
    def __init__(self, fps: int = None, window: str = None, n_fft: int = None,
            lcf: float = None, ucf: float = None,
            ldb: float = None, udb: float = None) -> None:
        """Create a new spectrum

        Args:
            fps:     Sample rate.
            window:  Name of window function.
            n_fft:   FFT length.
            lcf:     Lower cut-off frequency.
            ucf:     Upper cut-off frequency.
            ldb:     Lower dB boundary.
            udb:     Upper db_boundary.
        """
        self._params = container.SpectrumParams(fps, window, n_fft, lcf,
            ucf, ldb, udb)
        self._frqs = None
        self._bins = None

    def fit(self, inp: _Array) -> None:
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
    def d_frq(self):
        try:
            return self._frqs[1, 0] - self._frqs[0, 0]
        except TypeError:
            return None

    @property
    def abs(self) -> _Array:
        """Return trimmed and clipped magintude spectrum."""
        return self.__abs__()

    @property
    def bins(self) -> _Array:
        """Return rimmed FFT bins."""
        return self._bins

    @property
    def frqs(self) -> _Array:
        """Return trimmed frequency axis."""
        return self._frqs

    @property
    def params(self) -> container.SpectrumParams:
        """Return the parsed parameters."""
        return self._params

    @property
    def phase(self):
        """Return phase spectrum."""
        if self._bins is None:
            return None
        return _np.angle(self._bins)

    @property
    def power(self):
        """Retrun power spectrum."""
        return _np.square(self.__abs__())

    def centroid(self, power=True):
        return _np.multiply(self.abs(), self.frqs[:, None]).sum() / self.abs().sum()

    def plot(self, fmt='-'):
        import matplotlib.pyplot as plt
        plt.plot(self.frqs, self.abs, fmt)

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
        return self.length

    def __repr__(self):
        return 'Spectrum()'


class Spectrogram:
    """Compute spectrogram of an one-dimensional input array."""

    # pylint: disable=too-many-instance-attributes, too-many-arguments

    __slots__ = ('inp_size', 'fps', 'window', 'n_perseg', 'hop_size', 'n_overlap', 'n_fft',
                 'd_frq', 'd_time', 'times', 'frqs', 'bins', 'shape')

    def __init__(self, inp: _Array, fps: int, window: str, n_perseg: int, hop_size: int,
                 n_fft: int = None) -> None:
        """Compute a spectrogram of the input data.

        The input signal is segmented according to `n_perseg` and `hop_size`. To each
        segment FFT for real input is applied.

        If the segmentation parameters do not match the shape of the input array, the
        array is cropped.

        Args:
            inp:       Input signal.
            fps:       Sampling frequency of input signal.
            window:    Name of window function.
            n_perseg:  Number of samples per DFT.
            hop_size:  Number of samples to shift the window.
            n_fft:     Number of FFT bins.
        """
        self.inp_size = inp.size
        self.fps = fps
        self.window = window
        self.n_perseg = n_perseg
        self.hop_size = hop_size
        self.n_overlap = self.n_perseg - self.hop_size

        if n_fft is None:
            self.n_fft = self.n_perseg
        else:
            self.n_fft = n_fft

        self.d_frq = self.fps / self.n_fft
        self.d_time = self.hop_size / self.fps

        self.times = self._compute_time_axis(inp)
        self.frqs = _np.fft.rfftfreq(self.n_fft, 1.0/self.fps)
        self.bins = self._compute_spectrogram(inp)


    def _compute_time_axis(self, inp: _Array) -> _Array:
        """Compute the time axis of the spectrogram"""
        t_start = self.n_perseg / 2
        t_stop = inp.size - self.n_perseg / 2 + 1
        return _np.arange(t_start, t_stop, self.hop_size) / float(self.fps)

    def _compute_spectrogram(self, inp: _Array) -> _Array:
        """Core spectrogram computation.

        Args:
            inp (ndarray)    Input signal.
        """
        shp_x = (self.inp_size - self.n_overlap) // self.hop_size
        shp_y = self.n_perseg

        strd_x = self.hop_size * inp.strides[0]
        strd_y = inp.strides[0]

        inp_strided = _np.lib.stride_tricks.as_strided(inp, (shp_x, shp_y), (strd_x, strd_y))

        return fft(inp_strided, self.window, self.n_fft)

    def abs(self):
        """Return the magnitude spectrogram."""
        return self.__abs__()

    def power(self):
        """Return the power spectrogram."""
        return _np.square(self.__abs__())

    def centroid(self, power=True):
        if power is True:
            inp = self.power()
        else:
            inp = self.abs()

        return _features.spectral_centroid(inp.T, self.frqs)

    def flux(self, subband=False):
        flux = _features.spectral_flux(self.abs(), self.times)
        if subband is True:
            return flux
        return flux.sum(axis=0)

    def extract(self, cf_low: float = 50, cf_high: float = 10000):
        spctr = _features.spectral_shape(self.power(), self.frqs, cf_low, cf_high)
        prcpt = _features.perceptual_shape(self.abs(), self.frqs)
        tmpr = container.FeatureSpace(flux=self.flux())
        return container.FeatureSpace(spectral=spctr, perceptual=prcpt, temporal=tmpr)

    def params(self):
        return {'window': self.window, 'n_perseg': self.n_perseg,
                'hop_size': self.hop_size, 'n_fft': self.n_fft}

    def plot(self, cmap: str = 'nipy_spectral', log_frq: float = None,
             low: float = None, high: float = None, figsize: tuple = (14, 6),
             cbar: bool = True ) -> tuple:
        """Plot the spectrogram in dB scaling. The 0-frequency component
        is ommitted in plots.

        Args:
            cmap    (str)      Colormarp name.
            log_frq (float)    If None, plot the frequency axis linearly, else
                               plot it in log domain, centered on `log_frq` Hz.
            cbar    (bool)     Display a color scale if True.
            figsize (tuple)    Width and height of figure.

        Returns:
            Tuple    (fig, ax)
        """
        fig, ax = _plt.subplots(1, figsize=figsize)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')

        if low is None:
            low = 50

        if high is None:
            high = 16000

        low_idx = int(_np.floor(low/self.d_frq)) + 1
        high_idx = int(_np.floor(high/self.d_frq))

        vals = _sigtools.amp2db(self.abs()[low_idx:high_idx, :])
        frq_range = self.frqs[low_idx:high_idx]
        cmesh_frqs = _np.append(frq_range, frq_range[-1]+self.d_frq)
        if log_frq is not None:
            cmesh_frqs = _np.log2(cmesh_frqs/log_frq)

        cmesh_times = _np.append(self.times, self.times[-1]+self.d_time)
        cmesh = ax.pcolormesh(cmesh_times, cmesh_frqs, vals, cmap=cmap)

        if cbar:
            clr_bar = fig.colorbar(cmesh, ax=ax)
            clr_bar.set_label('db SPL')

        return fig, ax

    def __abs__(self):
        return _np.absolute(self.bins)


def stft(inp: _Array, fps: int, window: str = 'hanning', n_perseg: int = 512, hop_size: int = None,
         n_fft: int = None) -> Spectrogram:
    """Perform Short Time Fourier Transformation of `inp`

    `inp` is assumed to be an one-dimensional array of real values.

    Args:
        inp      (ndarray)    Input signal.
        fps      (int)        Sampling frequency of input signal.
        window   (str)        Name of window function.
        n_perseg (int)        Number of samples per DFT.
        hop_size (int)        Number of samples to shift the window.
        n_fft    (int)        Number of FFT bins.

    Returns:
        (Spectrogram)
    """

    # pylint: disable=too-many-arguments

    if hop_size is None:
        hop_size = n_perseg // 2

    return Spectrogram(inp, fps, window, n_perseg, hop_size, n_fft)

