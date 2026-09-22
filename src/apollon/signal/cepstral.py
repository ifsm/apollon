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
from . spectral import Spectrogram, Stft, full_scale_db
from .. typing import FloatArray, floatarray


ENERGY_FLOOR = 1e-10
"""Default floor of :func:`log_mel_energies`, in the units of the power passed.

That is -100 dB, and hence -100 dBFS for power at the default, calibrated STFT
scaling. The transforms do not use it: they convert ``floor_dbfs`` into the
units of their own power spectrum instead, see :func:`spectral.full_scale_db`.
"""


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
        floor:  Lower bound on the band energies, in the units of ``power``

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
                   lifter_gain: float = CepstrumParams().lifter_gain
                   ) -> FloatArray:
    """Compute cepstral coefficients from log band energies.

    The Discrete Cosine Transform decorrelates the log band energies and
    compacts their energy into the first few coefficients, of which the
    leading ``n_coefs`` are kept. ``scipy.fft.dct`` is used orthonormalized
    (``norm="ortho"``), which makes the transform unitary: the full set of
    coefficients carries the same energy as the band energies it came from,
    and so does not grow with the number of bands. This is also how
    ``librosa.feature.mfcc`` normalizes by default. The coefficients are then
    liftered, by default with the gain of ``CepstrumParams``. Pass
    ``lifter_gain=0.0`` to obtain the plain DCT.

    Following the convention of this package, the bands run along the first
    axis of ``log_energies``, as returned by :func:`log_mel_energies`, and so
    do the resulting coefficients.

    Args:
        log_energies:  Log band energies, shaped ``(n_filters, n_segments)``
        dct_type:      Type of the Discrete Cosine Transform, 1 to 4
        n_coefs:       Number of coefficients to keep. If ``None``, keep all
        lifter_gain:   Liftering parameter, defaulting to that of
                       ``CepstrumParams``. ``0.0`` leaves the coefficients
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

    coefs = floatarray(_spf.dct(log_energies, type=dct_type, axis=0, norm="ortho"))
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

    # pylint: disable = R0913
    def __init__(self, stft: StftParams, fb: TriangFilterSpec,
                 cepstrum: CepstrumParams | None = None,
                 preemphasis: float = 0.97,
                 floor_dbfs: float = -100.0) -> None:
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
            floor_dbfs:   Lowest band level retained, in dB relative to a
                          full-scale sinusoid. Band energies below it are
                          raised to it before the logarithm

        The floor bounds the dynamic range the logarithm sees. Far below the
        loudest part of a signal, band energies hold dither, quantization
        noise and leakage rather than signal, and the logarithm magnifies
        their fluctuation; the DCT would spread it over every coefficient.

        ``floor_dbfs`` is converted into the units of the STFT's power
        spectrum, so it denotes the same level whatever ``stft.norm`` and
        ``stft.single_sided`` are. Being fixed rather than taken from the
        data, it floors each segment independently of all others: cutting
        the signal into chunks, or adding a loud event elsewhere, leaves the
        coefficients of a segment unchanged. The reference is the full scale
        of the input, though. Peak-normalizing the signal first, as
        ``AudioFile.read(norm=True)`` does, turns it into a per-file
        reference and gives that independence up.

        The default suits 16-bit audio. It sits some 10 to 25 dB above the
        quantization noise of each band -- close enough to discard little
        signal, far enough that the noise cannot move a band by more than a
        fraction of a decibel. Material with a higher noise floor of its own,
        such as most acoustic recordings, warrants a higher floor.

        Raises:
            ValueError: If more cepstral coefficients are requested than the
                        filter bank has filters, if the filter bank cannot
                        be built on the frequency axis implied by ``stft``,
                        or if ``floor_dbfs`` is not finite
        """
        self._stft = Stft(fps=stft.fps, n_perseg=stft.n_perseg,
                          n_overlap=stft.n_overlap, window=stft.window,
                          n_fft=stft.n_fft, norm=stft.norm,
                          single_sided=stft.single_sided,
                          extend=stft.extend, pad=stft.pad)
        self._params = MfccParams(stft=self._stft.params, fb=fb,
                                  cepstrum=cepstrum or CepstrumParams(),
                                  preemphasis=preemphasis,
                                  floor_dbfs=floor_dbfs)
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
                 cepstrum: CepstrumParams | None = None,
                 floor_dbfs: float = -100.0) -> None:
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
            fb:          Specification of the triangular filter bank
            cepstrum:    Parameters of the cepstrum. If ``None``, use the
                         defaults of ``CepstrumParams``
            floor_dbfs:  Lowest band level retained, in dB relative to a
                         full-scale sinusoid. It is converted into the units
                         of each spectrogram's power by the spectrogram's own
                         params, see :class:`Mfcc` for the rationale

        Raises:
            ValueError: If more cepstral coefficients are requested than the
                        filter bank has filters, or if ``floor_dbfs`` is not
                        finite
        """
        self._params = CepstralParams(fb=fb, floor_dbfs=floor_dbfs,
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
                            floor_dbfs=self._params.floor_dbfs,
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
    floor = 10**((params.floor_dbfs + full_scale_db(params.stft)) / 10)
    energies = log_mel_energies(sxx.power, fbank, floor)
    coefs = cepstral_coefs(energies, params.cepstrum.dct_type,
                           params.cepstrum.n_coefs,
                           params.cepstrum.lifter_gain)
    return MelCepstrogram(params, coefs, energies, fbank, sxx.times, sxx.frqs)
