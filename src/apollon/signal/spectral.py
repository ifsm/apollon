"""
Spectral transforms
====================
"""

from abc import ABC, abstractmethod
from typing import Any, cast

import matplotlib.pyplot as _plt
import numpy as np
import scipy.signal as _sps

from apollon.segment import ArraySegmentation, Segments
from apollon.segment.models import SegmentationParams

from . models import (DftParams, Normalization, StftParams,
                      SpectralTransformParams)
from .. typing import FloatArray, IntArray, ComplexArray, floatarray
from .. signal import features


def fft(sig: FloatArray, window: str | None = None, n_fft: int | None = None,
        norm: Normalization | None = 'amplitude',
        single_sided: bool = True) -> ComplexArray:
    """Compute the Discrete Fouier Transform for real input

    This is a simple wrapper around ``numpy.fft.rfft``. Input signal must
    be two-dimensional. FTT is performed along the rows.

    ``norm`` selects the scaling convention:

    ============  =========================================================
    ``None``      Leave the bins as ``rfft`` returns them
    ``'ortho'``   Divide by ``sqrt(n_fft)``, making the transform unitary
    ``amplitude`` Divide by the window's coherent gain, so that a sinusoid
                  reads its own amplitude
    ============  =========================================================

    ``rfft`` drops the negative half of the spectrum, where a real sinusoid
    keeps half of its amplitude. If ``single_sided`` is ``True``, the bins
    that lost a partner there are scaled up to account for it, by ``sqrt(2)``
    under ``'ortho'``, which conserves energy, and by two otherwise, which
    conserves amplitude. Note that ``'amplitude'`` without ``single_sided``
    hence reads a sinusoid of amplitude ``A`` as ``A/2``.

    Args:
        sig:           Two-dimensional input array
        n_fft:         FFT length in samples
        window:        Name of window function
        norm:          Scaling convention
        single_sided:  If ``True``, account for the discarded negative half
                       of the spectrum

    Returns:
        FFT bins

    Raises:
        ValueError:  If ``sig`` is not two-dimensional, if ``norm`` is
            not one of the above, or if ``n_fft`` is less than the length
            of ``sig``, which would crop the signal
    """
    if sig.ndim != 2:
        raise ValueError(f'Input array has {sig.ndim} dimensions. However,'
                         ' ``fft`` expects two-dimensional array.')
    if norm not in (None, 'ortho', 'amplitude'):
        raise ValueError(f'Invalid norm value {norm!r}; should be None,'
                         ' "ortho" or "amplitude".')
    n_sig = sig.shape[0]
    if n_fft is None:
        n_fft = n_sig
    if n_fft < n_sig:
        raise ValueError(f'n_fft ({n_fft}) is less than the {n_sig} samples '
                         'of the signal, and would crop it.')

    if window is None:
        window = 'rect'

    win = np.expand_dims(_sps.get_window(window, n_sig), 1)
    bins = np.fft.rfft(sig*win, n_fft, axis=0,
                       norm='ortho' if norm == 'ortho' else 'backward')

    if norm == 'amplitude':
        bins /= abs(win.sum())

    if single_sided:
        bins[_paired_bins(n_fft)] *= np.sqrt(2) if norm == 'ortho' else 2

    return bins


def _paired_bins(n_fft: int) -> slice:
    """Select the bins that have a partner in the negative half spectrum

    ``numpy.fft.rfft`` discards the negative frequencies, which hold half the
    amplitude of each real sinusoid. Doubling the remaining bins restores it,
    but only for those that actually lost a partner. The zeroth bin never has
    one, and neither has the Nyquist bin, which ``rfft`` returns as the last
    bin for even ``n_fft`` only. For odd ``n_fft`` there is no Nyquist bin and
    the last bin is paired like any other.

    Args:
        n_fft:  FFT length in samples

    Returns:
        Index of the paired bins along the frequency axis
    """
    return slice(1, -1 if n_fft % 2 == 0 else None)


def full_scale_db(params: StftParams) -> float:
    """Compute the level at which full scale appears in the power spectrum.

    ``Spectrogram.power`` carries the scale of the transform that produced
    it. That scale is set by ``norm`` and ``single_sided``, and depends on the
    window and the FFT length besides. This returns where a full-scale
    sinusoid -- unit amplitude, centred on a paired bin -- lands on that
    scale, so that a level stated in dB relative to full scale can be
    converted into the units of the power spectrum by adding it.

    Under the default scaling, ``norm='amplitude'`` with
    ``single_sided=True``, the result is 0 dB: power is then calibrated to
    full scale already.

    Args:
        params:  Parameters of the Short Time Fourier Transform

    Returns:
        Level of a full-scale sinusoid in dB, in the units of the power
        spectrum.
    """
    win_sum = abs(_sps.get_window(params.window or 'rect', params.n_perseg).sum())
    n_fft = params.n_perseg if params.n_fft is None else params.n_fft

    peak = win_sum / 2      # |rfft| of a unit sinusoid on a paired bin
    if params.norm == 'amplitude':
        peak /= win_sum
    elif params.norm == 'ortho':
        peak /= np.sqrt(n_fft)
    if params.single_sided:
        peak *= np.sqrt(2) if params.norm == 'ortho' else 2
    return float(20 * np.log10(peak))


class TransformResult(ABC):
    """Base class for transformation results"""
    def __init__(self, bins: ComplexArray) -> None:
        self._bins = bins
        self._params: DftParams
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
        return cast(FloatArray, np.fft.rfftfreq(self._n_fft,
                               1.0/self._params.fps).reshape(-1, 1))

    @property
    @abstractmethod
    def params(self) -> DftParams:
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
    def ms_power(self) -> FloatArray:
        """Compute the mean-square power per bin

        ``power`` carries the scale of the transform. Under the default
        ``norm='amplitude'`` it reads the amplitude of a sinusoid, but summed
        over bins it overcounts by the equivalent noise bandwidth of the
        window. This undoes the scaling ``fft`` applied and normalizes by the
        power of the window instead, so that the bins of each spectrum sum to
        the window-weighted mean square of its frame, whatever ``norm``,
        ``single_sided``, window, and FFT length. It is hence the quantity to
        sum over frequency bands. For a signal in Pa, it is in Pa².
        """
        win = _sps.get_window(self._params.window or 'rect', self._inp_size)
        n_fft = self._n_fft
        norm = self._params.norm

        pwr = self.power
        if norm == 'amplitude':
            pwr *= win.sum()**2
        elif norm == 'ortho':
            pwr *= n_fft

        paired = _paired_bins(n_fft)
        if self._params.single_sided:
            pwr[paired] /= 2 if norm == 'ortho' else 4
        pwr[paired] *= 2
        return floatarray(pwr / (n_fft * np.sum(np.square(win))))

    @property
    def centroid(self) -> FloatArray:
        """Compute spectral centroid"""
        return features.spectral_centroid(self.frqs, self.power)

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
        return np.asarray(self._bins[key]).astype(np.complex128)

    def __len__(self) -> int:
        return int(self._bins.shape[0])


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
        return int(self._bins.shape[1])

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
                 n_fft: int | None = None,
                 norm: Normalization | None = 'amplitude',
                 single_sided: bool = True) -> None:
        """Create a new spectrum

        Args:
            fps:           Sample rate
            window:        Name of window function
            n_fft:         FFT length
            norm:          Scaling convention, see ``fft``
            single_sided:  If ``True``, account for the discarded negative
                           half of the spectrum, see ``fft``
        """
        super().__init__()
        self._params: DftParams = DftParams(fps=fps, window=window, n_fft=n_fft,
                                            norm=norm, single_sided=single_sided)

    def transform(self, data: FloatArray) -> Spectrum:
        """Transform ``data`` to spectral domain."""
        bins = fft(data, self.params.window, self.params.n_fft,
                   norm=self.params.norm,
                   single_sided=self.params.single_sided)
        return Spectrum(self.params, bins, data.shape[0])

    @property
    def params(self) -> DftParams:
        return self._params


class Stft(SpectralTransform):
    """Short Time Fourier Transform of AudioFile."""
    def __init__(self, fps: int, n_perseg: int, n_overlap: int,
                 window: str | None = None,
                 n_fft: int | None = None,
                 norm: Normalization | None = 'amplitude',
                 single_sided: bool = True,
                 extend: bool = True, pad: bool = True) -> None:
        # pylint: disable = R0913
        """Create a new spectrogram.

        Args:
            fps:           Sample rate
            n_perseg:      Samples per segment
            n_overlap:     Number of overlapping samples per segment
            window:        Name of window function
            n_fft:         FFT length
            norm:          Scaling convention, see ``fft``
            single_sided:  If ``True``, account for the discarded negative
                           half of the spectrum, see ``fft``
            extend:        If ``True``, extend the signal at both ends
            pad:           If ``True``, pad the last segment with zeros
        """
        super().__init__()
        self._params: StftParams = StftParams(fps=fps, window=window, n_fft=n_fft,
                                    norm=norm, single_sided=single_sided,
                                    n_perseg=n_perseg, n_overlap=n_overlap,
                                    extend=extend, pad=pad)
        self._cutter = ArraySegmentation(self.params.n_perseg, self.params.n_overlap,
                                         self.params.extend, self.params.pad)

    def transform(self, data: FloatArray) -> Spectrogram:
        """Transform ``data`` to spectral domain"""
        segs = self._cutter.transform(data)
        bins = fft(segs.data, self.params.window, self.params.n_fft,
                   norm=self.params.norm,
                   single_sided=self.params.single_sided)
        return Spectrogram(self._params, bins, segs.params.n_perseg)

    @property
    def params(self) -> StftParams:
        return self._params


class StftSegments(SpectralTransform):
    """Short Time Fourier Transform on already segmented audio"""
    def __init__(self, fps: int, seg_params: SegmentationParams, window: str | None = None,
                 n_fft: int | None = None,
                 norm: Normalization | None = 'amplitude',
                 single_sided: bool = True) -> None:
        """Create a new ``Spectrogram`` from ``Segments``

        Args:
            fps:           Sample rate
            seg_params:    Parameters of the segmentation behind the input
            window:        Name of window function
            n_fft:         FFT length
            norm:          Scaling convention, see ``fft``
            single_sided:  If ``True``, account for the discarded negative
                           half of the spectrum, see ``fft``
        """
        super().__init__()
        self._seg_params = seg_params
        self._params: StftParams = StftParams(fps=fps, window=window,
                                              n_fft=n_fft, norm=norm,
                                              single_sided=single_sided,
                                              **seg_params.model_dump())

    def transform(self, data: Segments) -> Spectrogram:
        """Transform ``data`` to spectral domain

        Args:
            data:  Segments cut with the parameters given at construction

        Returns:
            Spectrogram of ``data``

        Raises:
            ValueError: If ``data`` was segmented with other parameters,
                which would give the spectrogram a wrong time axis.
        """
        if data.params != self._seg_params:
            raise ValueError(f'``data`` was segmented with {data.params!r}, '
                             f'but this transform expects {self._seg_params!r}.')
        bins = fft(data.data, self._params.window, self._params.n_fft,
                   norm=self._params.norm,
                   single_sided=self._params.single_sided)
        return Spectrogram(self._params, bins, data.params.n_perseg)

    @property
    def params(self) -> StftParams:
        return self._params
