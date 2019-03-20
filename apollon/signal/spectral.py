"""spectral.py    (c) Michael Blaß 2016

Provide easy access to frequency spectra obtained by the DFT.

Classes:
    _Spectrum_Base      Utility class
    _Spectrum           Representation of a frequency spectrum

Functions:
    fft                 Easy to use discrete fourier transform
"""


import matplotlib.pyplot as _plt
import numpy as _np
from scipy.signal import get_window as _get_window

from . import tools as _tools
from . import features as _features
from .. types import Array as _Array


class _Spectrum_Base:
    def __abs__(self):
        return _np.absolute(self.bins)

    def __add__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins + other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins + other, sr=self.sr,
                             n=self.n, window=self.window)

    def __radd__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins + other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins + other, sr=self.sr,
                             n=self.n, window=self.window)

    def __sub__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins - other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins - other, sr=self.sr,
                             n=self.n, window=self.window)

    def __rsub__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return_Spectrum(self.bins - other.bins, sr=self.sr,
                                n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins - other, sr=self.sr,
                             n=self.n, window=self.window)

    def __mul__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins * other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins * other, sr=self.sr,
                             n=self.n, window=self.window)

    def __rmul__(self, other):
        if isinstance(other, _Spectrum):
            if self.sr == other.sr and self.n == other.n:
                return _Spectrum(self.bins * other.bins, sr=self.sr,
                                 n=self.n, window=self.window)
            else:
                raise ValueError('Spectra not compatible.')
        else:
            return _Spectrum(self.bins * other, sr=self.sr,
                             n=self.n, window=self.window)


class _Spectrum(_Spectrum_Base):
    def __init__(self, spectral_data, sr, n, window=None, *args, **kwargs):
        self.bins = spectral_data
        self.sr = sr
        self.n = n
        self.window = window
        self.freqs = _np.fft.rfftfreq(self.n, 1/self.sr)

    def __getitem__(self, key):
        return self.bins[key]

    def __len__(self):
        return self.length

    def __repr__(self):
        return 'Spectrum(bins={}, sr={}, n={}, window={})'.format(self.bins,
                                                                  self.sr,
                                                                  self.n,
                                                                  self.window)

    def centroid(self):
        """Return spectral centroid."""
        powspc = self.power()
        return (self.freqs * powspc).sum() / powspc.sum()

    def __abs__(self):
        return _np.absolute(self.bins)

    def abs(self):
        """Return magnitude spectrum."""
        return self.__abs__()

    def power(self):
        """Retrun power spectrum."""
        return _np.square(self.abs())

    def phase(self):
        """Return phase spectrum."""
        return _np.angle(self.bins)

    def plot(self, db=True, fmt='-', logfreq=False):
        """Plot magnitude spectrum.

        Params:
            db         (bool) set True to plot amplitudes db-scaled.
            fmt        (str) matplotlib linestyle string.
            logfreq    (bool) set True to log-scale .x axis.
        """
        fig, ax = _plt.subplots(1)

        if logfreq:
            plot_function = ax.semilogx
        else:
            plot_function = ax.plot

        if db:
            plot_function(self.freqs, 20*_np.log10(self.mag()),
                          fmt, lw=2, alpha=.7)
            ax.set_ylabel(r'Amplitude [dB]')
        else:
            plot_function(self.freqs, self.mag(),
                          fmt, lw=2, alpha=.7)
            ax.set_ylabel(r'Amplitude')

        ax.set_xlabel(r'Frequency [Hz]')
        ax.grid()

        if not _plt.isinteractive():
            fig.show()


def fft(sig, window=None, n_fft=None):
    """Return the Discrete Fourier Transform for real input.

    Params:
        sig    (array-like)    Input time domain signal
        fs     (int)           Sample rate
        n_fft  (int)           FFT length
        window (str)           Name of window function

    Returns:
        (ndarray) FFT bins.
    """
    sig = _np.atleast_2d(sig).astype('float64')
    n_sig = sig.shape[-1]

    if n_fft is None:
        n_fft = n_sig

    if window is not None:
        sig = _np.multiply(sig, _get_window(window, n_sig))

    bins = _np.fft.rfft(sig, n_fft)
    bins = _np.divide(bins, float(n_fft))

    if n_fft % 2 != 0:
        bins = _np.multiply(bins[:, :-1], 2.0)
    else:
        bins = _np.multiply(bins, 2.0)

    return bins.squeeze()


class Spectrogram:
    def __init__(self, inp:_Array, fs:int, window:str, n_perseg:int, hop_size:int) -> None:
        """Compute a spectrogram of the input data.

        The input signal is segmented according to `n_perseg` and `hop_size`. To each
        segment FFT for real input is applied.

        If the segmentation parameters do not match the shape of the input array, the
        array is cropped.

        Args:
            inp      (np.ndarray)    Input signal.
            fs       (int)           Sampling frequency of input signal.
            window   (str)           Name of window function.
            n_perseg (int)           Number of samples per DFT.
            hop_size (int)           Number of samples to shift the window.

        Returns:
            (Spectrogram)
        """
        self.inp_size = inp.size
        self.fs = fs
        self.window = window
        self.n_perseg = n_perseg
        self.hop_size = hop_size
        self.n_overlap = self.n_perseg - self.hop_size

        self.bins = None
        self.frqs = None
        self.times = None
        self._compute_spectrogram(inp)

    def _compute_spectrogram(self, inp):
        shp_x = (self.inp_size - self.n_overlap ) // self.hop_size
        shp_y = self.n_perseg

        strd_x = self.hop_size * inp.strides[0]
        strd_y = inp.strides[0]

        inp_strided = _np.lib.stride_tricks.as_strided(inp, (shp_x, shp_y), (strd_x, strd_y))

        self.bins = fft(inp_strided, self.window)
        self.bins = _np.transpose(self.bins)
        self.frqs = _np.fft.rfftfreq(self.n_perseg, 1/self.fs)

        t_start = shp_y / 2
        t_stop = inp.size - shp_y / 2 + 1
        self.times = _np.arange(t_start, t_stop, self.hop_size) / float(self.fs)

    def abs(self):
        return self.__abs__()

    def power(self):
        return _np.square(self.__abs__())

    def centroid(self, power=True):
        if power is True:
            inp = self.power()
        else:
            inp = self.abs()

        return _features.spectral_centroid(inp.T, self.frqs)

    def flux(self, subband=False):
        flux = _features.spectral_flux(self.abs())
        if subband is True:
            return flux
        return flux.sum(axis=0)


    def plot(self, cmap:str = 'nipy_spectral', cbar:bool = True,
             figsize:tuple = (8, 4)) -> tuple:
        """Plot the spectrogram in dB scaling.

        Args:
            cmap    (str)      Colormarp name.
            cbar    (bool)     Display a color scale if True.
            figsize (tuple)    Width and height of figure.

        Returns:
            Tuple    (fig, ax)
        """
        fig, ax = _plt.subplots(1, figsize=figsize)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        cmesh = ax.pcolormesh(self.times, self.frqs, _amp2db(self.abs()), cmap=cmap)

        if cbar:
            cb = fig.colorbar(cmesh, ax=ax)
            cb.set_label('db SPL')

        return fig, ax

    def __abs__(self):
        return _np.absolute(self.bins)


def stft(inp: _Array, fs:int, window:str = 'hanning', n_perseg:int = 512, hop_size:int = 256):
    """Perform Short Time Fourier Transformation of `inp`

    `inp` is assumed to be an one-dimensional array of real values.

    Args:
        inp      (np.ndarray)    Input signal.
        fs       (int)           Sampling frequency of input signal.
        window   (str)           Name of window function.
        n_perseg (int)           Number of samples per DFT.
        hop_size (int)           Number of samples to shift the window.

    Returns:
        (Spectrogram)
    """
    return Spectrogram(inp, fs, window, n_perseg, hop_size)
