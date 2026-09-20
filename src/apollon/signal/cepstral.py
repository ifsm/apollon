"""
Cepstral transforms
===================

Mel-frequency cepstral coefficients and the steps they are built from.

Cepstral analysis is a *post-spectral* stage: it consumes a power spectrum
and returns coefficients indexed by their own ordinal, not by frequency. The
types defined here therefore mirror the parameter/result pattern of
``apollon.signal.spectral`` without sharing its class hierarchy, whose
members (``phase``, ``frqs``, ``centroid``, ...) carry no meaning in the
cepstral domain.

Two transforms are provided. :class:`Mfcc` owns the whole chain from the
time-domain signal and is the one to reach for by default. Pass an existing
``Spectrogram`` to :class:`MfccSpectrogram` instead to reuse a Short Time
Fourier Transform that has already been computed -- note that pre-emphasis
acts on the signal and is hence out of that transform's reach.
"""

import numpy as np
import scipy.fft as _spf

from . import filter as _filter
from . models import (CepstralParams, CepstrumParams, MfccParams, StftParams,
                      TriangFilterSpec)
from . spectral import Spectrogram, Stft
from .. typing import FloatArray, floatarray


ENERGY_FLOOR = 1e-10
"""Smallest band energy considered non-zero, see :func:`log_mel_energies`."""


def log_mel_energies(power: FloatArray, fbank: FloatArray,
                     floor: float = ENERGY_FLOOR) -> FloatArray:
    """Compute the log energy per filter of a filter bank.

    Each filter of ``fbank`` is applied to every column of ``power``, and the
    resulting band energies are converted to decibels as
    ``10*log10(energy)``.

    Band energies of zero -- a silent frame, or a filter whose entire support
    falls in a spectral null -- have no logarithm. They are clamped to
    ``floor`` beforehand, which maps them to a large negative value instead
    of ``-inf`` and keeps the subsequent DCT finite.

    Following the convention of this package, frequency runs along the first
    axis of ``power``, as in ``spectral.Spectrogram.power``. The filters of
    ``fbank`` run along its first axis, as returned by
    ``filter.triangular_filter_bank``.

    Args:
        power:  Power spectrum, shaped ``(n_frqs, n_segments)``
        fbank:  Filter bank, shaped ``(n_filters, n_frqs)``
        floor:  Lower bound on the band energies

    Returns:
        Log band energies in dB, shaped ``(n_filters, n_segments)``.

    Raises:
        ValueError: If ``floor`` is not positive, or if the filters of
            ``fbank`` are not defined on as many frequencies as ``power``
            holds.
    """
    if floor <= 0:
        raise ValueError("``floor`` is not positive")

    if fbank.shape[-1] != power.shape[0]:
        raise ValueError(f"Filter bank is defined on {fbank.shape[-1]} "
                         f"frequencies, but ``power`` holds {power.shape[0]}.")

    return floatarray(10 * np.log10(np.maximum(fbank @ power, floor)))


def cepstral_coefs(log_energies: FloatArray, dct_type: int = 2,
                   n_coefs: int | None = None,
                   lifter_gain: float = 0.0) -> FloatArray:
    """Compute cepstral coefficients from log band energies.

    The Discrete Cosine Transform decorrelates the log band energies and
    compacts their energy into the first few coefficients, of which the
    leading ``n_coefs`` are kept. ``scipy.fft.dct`` is used unnormalized, so
    the coefficients scale with the number of bands.

    Following the convention of this package, the bands run along the first
    axis of ``log_energies``, as returned by :func:`log_mel_energies`, and so
    do the resulting coefficients.

    Args:
        log_energies:  Log band energies, shaped ``(n_filters, n_segments)``
        dct_type:      Type of the Discrete Cosine Transform, 1 to 4
        n_coefs:       Number of coefficients to keep. If ``None``, keep all
        lifter_gain:   Liftering parameter. ``0.0`` leaves the coefficients
                       unchanged, see ``filter.lifter``

    Returns:
        Cepstral coefficients, shaped ``(n_coefs, n_segments)``.

    Raises:
        ValueError: If ``n_coefs`` is less than one or exceeds the number of
            bands in ``log_energies``, or if ``lifter_gain`` is negative.
    """
    n_bands = log_energies.shape[0]
    if n_coefs is not None:
        if n_coefs < 1:
            raise ValueError("``n_coefs`` is less than one")
        if n_coefs > n_bands:
            raise ValueError(f"Requested {n_coefs} cepstral coefficients from "
                             f"{n_bands} bands. The cepstrum cannot hold more "
                             "coefficients than there are bands.")

    coefs = floatarray(_spf.dct(log_energies, type=dct_type, axis=0))
    if n_coefs is not None:
        coefs = coefs[:n_coefs]
    return _filter.lifter(coefs, lifter_gain)


class MelCepstrogram:
    """Result of a cepstral transform"""

    # pylint: disable = R0913
    def __init__(self, params: MfccParams, coefs: FloatArray,
                 energies: FloatArray, filter_bank: FloatArray,
                 times: FloatArray, frqs: FloatArray) -> None:
        """Cepstral coefficients with time axis and the stages they came from.

        Args:
            params:       Parameters of the transform that produced the
                          coefficients
            coefs:        Cepstral coefficients, shaped
                          ``(n_coefs, n_segments)``
            energies:     Log band energies before the DCT, shaped
                          ``(n_filters, n_segments)``
            filter_bank:  Filter bank applied to the power spectrum, shaped
                          ``(n_filters, n_frqs)``
            times:        Time axis of the underlying spectrogram in seconds
            frqs:         Frequency axis the filter bank is defined on
        """
        self._params = params
        self._coefs = coefs
        self._log_mel_energies = energies
        self._filter_bank = filter_bank
        self._times = times
        self._frqs = frqs

    @property
    def coefs(self) -> FloatArray:
        """Cepstral coefficients, shaped ``(n_coefs, n_segments)``"""
        return self._coefs

    @property
    def params(self) -> MfccParams:
        """Parameters of the transform that produced the coefficients"""
        return self._params

    @property
    def log_mel_energies(self) -> FloatArray:
        """Log band energies in dB, the input of the DCT"""
        return self._log_mel_energies

    @property
    def filter_bank(self) -> FloatArray:
        """Filter bank applied to the power spectrum"""
        return self._filter_bank

    @property
    def times(self) -> FloatArray:
        """Time axis in seconds, one entry per segment"""
        return self._times

    @property
    def frqs(self) -> FloatArray:
        """Frequency axis the filter bank is defined on"""
        return self._frqs

    @property
    def n_coefs(self) -> int:
        """Number of cepstral coefficients per segment"""
        return int(self._coefs.shape[0])

    @property
    def n_segments(self) -> int:
        """Number of segments"""
        return int(self._coefs.shape[1])

    def __getitem__(self, key: int) -> FloatArray:
        return floatarray(self._coefs[key])

    def __len__(self) -> int:
        return int(self._coefs.shape[0])

    def __repr__(self) -> str:
        return f'MelCepstrogram({self._params})'


class Mfcc:
    """Mel-frequency cepstral coefficients of a signal"""

    def __init__(self, stft: StftParams, fb: TriangFilterSpec,
                 cepstrum: CepstrumParams | None = None,
                 preemphasis: float = 0.97) -> None:
        """Transform a signal to Mel-frequency cepstral coefficients.

        The transform owns the whole chain: pre-emphasis of the signal, the
        Short Time Fourier Transform, the filter bank, and the cepstrum. The
        filter bank is built once here, so a filter specification that the
        frequency resolution of ``stft`` cannot support fails at construction
        rather than during a transform.

        Following the convention of ``spectral.Stft``, the input signal is
        single-channel and shaped ``(n_frames, 1)``.

        Args:
            stft:         Parameters of the Short Time Fourier Transform
            fb:           Specification of the triangular filter bank
            cepstrum:     Parameters of the cepstrum. If ``None``, use the
                          defaults of ``CepstrumParams``
            preemphasis:  Pre-emphasis coefficient applied to the signal.
                          ``0.0`` disables it

        Raises:
            ValueError: If more cepstral coefficients are requested than the
                        filter bank has filters, or if the filter bank cannot
                        be built on the frequency axis implied by ``stft``
        """
        self._stft = Stft(fps=stft.fps, n_perseg=stft.n_perseg,
                          n_overlap=stft.n_overlap, window=stft.window,
                          n_fft=stft.n_fft, extend=stft.extend, pad=stft.pad)
        self._params = MfccParams(stft=self._stft.params, fb=fb,
                                  cepstrum=cepstrum or CepstrumParams(),
                                  preemphasis=preemphasis)
        self._fbank = _build_fbank(_rfftfreq(self._stft.params), fb)

    def transform(self, data: FloatArray) -> MelCepstrogram:
        """Transform ``data`` to the cepstral domain

        Args:
            data:  Single-channel signal, shaped ``(n_frames, 1)``

        Returns:
            Cepstral coefficients and the stages they came from
        """
        coef = self._params.preemphasis
        if coef:
            data, _ = _filter.preemphasis(data, coef)
        return _assemble(self._stft.transform(data), self._fbank, self._params)

    @property
    def params(self) -> MfccParams:
        """Return parameters"""
        return self._params


class MfccSpectrogram:
    """Mel-frequency cepstral coefficients of an existing ``Spectrogram``"""

    def __init__(self, fb: TriangFilterSpec,
                 cepstrum: CepstrumParams | None = None) -> None:
        """Transform a spectrogram to Mel-frequency cepstral coefficients.

        Use this transform to reuse a Short Time Fourier Transform that has
        already been computed. It acts on the power spectrum only and can
        hence not apply pre-emphasis, which is a filter on the time-domain
        signal. Pre-emphasize the signal with ``filter.preemphasis`` before
        transforming it, or use :class:`Mfcc`, which does so itself. The
        ``preemphasis`` field of the resulting ``MfccParams`` is ``None``,
        recording that this transform does not know whether the signal behind
        the spectrogram was pre-emphasized.

        The filter bank depends on the frequency axis of the spectrogram and
        is built on the first transform, then reused as long as the axis
        stays the same.

        Args:
            fb:        Specification of the triangular filter bank
            cepstrum:  Parameters of the cepstrum. If ``None``, use the
                       defaults of ``CepstrumParams``

        Raises:
            ValueError: If more cepstral coefficients are requested than the
                        filter bank has filters
        """
        self._params = CepstralParams(fb=fb,
                                      cepstrum=cepstrum or CepstrumParams())
        self._frqs: FloatArray | None = None
        self._fbank: FloatArray | None = None

    def transform(self, data: Spectrogram) -> MelCepstrogram:
        """Transform ``data`` to the cepstral domain

        Args:
            data:  Spectrogram to take the power spectrum from

        Returns:
            Cepstral coefficients and the stages they came from

        Raises:
            ValueError: If the filter bank cannot be built on the frequency
                axis of ``data``
        """
        fbank = self._fbank
        if (fbank is None or self._frqs is None
                or not np.array_equal(self._frqs, data.frqs)):
            fbank = _build_fbank(data.frqs, self._params.fb)
            self._fbank = fbank
            self._frqs = data.frqs
        params = MfccParams(stft=data.params, fb=self._params.fb,
                            cepstrum=self._params.cepstrum, preemphasis=None)
        return _assemble(data, fbank, params)

    @property
    def params(self) -> CepstralParams:
        """Return parameters"""
        return self._params


def _rfftfreq(params: StftParams) -> FloatArray:
    """Compute the frequency axis a ``Spectrogram`` from ``params`` will have.

    ``spectral.Stft`` passes ``n_perseg`` as the input size of its result, so
    an unset ``n_fft`` falls back to ``n_perseg``.

    Args:
        params:  Parameters of the Short Time Fourier Transform

    Returns:
        Frequency axis in Hz, shaped ``(n_frqs, 1)``.
    """
    n_fft = params.n_perseg if params.n_fft is None else params.n_fft
    return floatarray(np.fft.rfftfreq(n_fft, 1.0/params.fps).reshape(-1, 1))


def _build_fbank(frqs: FloatArray, fb: TriangFilterSpec) -> FloatArray:
    """Build the triangular filter bank described by ``fb`` on ``frqs``.

    Args:
        frqs:  Frequency axis in Hz
        fb:    Specification of the filter bank

    Returns:
        Filter bank, shaped ``(n_filters, n_frqs)``.
    """
    return _filter.triangular_filter_bank(frqs, fb.low, fb.high, fb.n_filters,
                                          fb.scale)


def _assemble(sxx: Spectrogram, fbank: FloatArray,
              params: MfccParams) -> MelCepstrogram:
    """Run the cepstral chain on ``sxx`` and collect the result.

    Args:
        sxx:     Spectrogram to take the power spectrum from
        fbank:   Filter bank, defined on the frequency axis of ``sxx``
        params:  Parameters to attach to the result

    Returns:
        Cepstral coefficients and the stages they came from.
    """
    energies = log_mel_energies(sxx.power, fbank)
    coefs = cepstral_coefs(energies, params.cepstrum.dct_type,
                           params.cepstrum.n_coefs,
                           params.cepstrum.lifter_gain)
    return MelCepstrogram(params, coefs, energies, fbank, sxx.times, sxx.frqs)
