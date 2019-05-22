# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

import json as _json
import csv as _csv
import sys as _sys
from typing import Any, Dict, List, Tuple

import numpy as _np
from scipy.signal import hilbert as _hilbert

from .. import segment as _segment
from .. tools import array2d_fsum
from .. types import Array as _Array
from .. io import ArrayEncoder
from .  import critical_bands as _cb
from .  roughness import roughness
from .  tools import trim_spectrogram


def spectral_centroid(inp: _Array, frqs: _Array) -> _Array:
    """Estimate the spectral centroid frequency.

    Calculation is applied to the second axis. One-dimensional
    arrays will be promoted.

    Args:
        inp:   Input data. Each row is assumend FFT bins scaled by `freqs`.
        frqs:  One-dimensional array of FFT frequencies.

    Returns:
        Array of Spectral centroid frequencies.
    """
    inp = _np.atleast_2d(inp).astype('float64')

    weighted_nrgy = _np.multiply(inp, frqs).sum(axis=1).squeeze()
    total_nrgy = inp.sum(axis=1).squeeze()
    total_nrgy[total_nrgy == 0.0] = 1.0

    return _np.divide(weighted_nrgy, total_nrgy)


def spectral_flux(inp: _Array, delta: float = 1.0) -> _Array:
    """Estimate the spectral flux

    Args:
        inp:    Input data. Each row is assumend FFT bins.
        delta:  Sample spacing.

    Returns:
        Array of Spectral flux.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    return _np.maximum(_np.gradient(inp, delta, axis=-1), 0).squeeze()


def spectral_shape(inp: _Array, frqs: _Array, low: float = 50, high: float = 16000):
    """Compute low-level spectral shape descriptors.

    This function computes the first four central moments of
    the input spectrum. If input is two-dimensional, the first
    axis is assumed to represent frequency.

    The central moments are:
        - Spectral Centroid (SC)
        - Spectral Spread (SSP),
        - Spectral Skewness (SSK)
        - Spectral Kurtosis (SKU).

    Spectral Centroid represents the center of gravity of the spectrum.
    It correlates well with the perception of auditory brightness.

    Spectral Spread is a measure for the frequency deviation around the
    centroid.

    Spectral Skewness is a measure of spectral symmetry. For values of
    SSK = 0 the spectral distribution is exactly symmetric. SSK > 0 indicates
    more power in the frequency domain below the centroid and vice versa.

    Spectral Kurtosis is a measure of flatness. The lower the value, the faltter
    the distribution.

    Args:
        inp:   Input spectrum or spectrogram.
        frqs:  Frequency axis.
        low:  Lower cutoff frequency.
        high:  Upper cutoff frequency.

    Returns:
        Spectral centroid, spread, skewness, and kurtosis.
    """
    if inp.ndim < 2:
        inp = inp[:, None]

    vals, frqs = trim_spectrogram(inp, frqs, 50, 16000)

    total_nrgy = array2d_fsum(vals)
    total_nrgy[total_nrgy == 0.0] = 1.0    # Total energy is zero iff input signal is all zero.
                                           # Replace these bin values with 1, so division by
                                           # total energy will not lead to nans.

    centroid = frqs @ vals / total_nrgy
    deviation = frqs[:, None] - centroid

    spread = array2d_fsum(_np.power(deviation, 2) * vals)
    skew   = array2d_fsum(_np.power(deviation, 3) * vals)
    kurt   = array2d_fsum(_np.power(deviation, 4) * vals)

    spread = _np.sqrt(spread / total_nrgy)
    skew   = skew / total_nrgy / _np.power(spread, 3)
    kurt   = kurt / total_nrgy / _np.power(spread, 4)

    return FeatureSpace(centroid=centroid, spread=spread, skewness=skew, kurtosis=kurt)


def log_attack_time(inp: _Array, fs: int, ons_idx: _Array, wlen:float=0.05) -> _Array:
    """Estimate the attack time of each onset and return its logarithm.

    This function estimates the attack time as the duration between the
    onset and the local maxima of the magnitude of the Hilbert transform
    of the local window.

    Args:
        inp:      Input signal.
        fs:       Sampling frequency.
        ons_idx:  Sample indices of onsets.
        wlen:     Local window length in samples.

    Returns:
        Logarithm of the attack time.
    """
    wlen = int(fs * wlen)
    segs = _segment.by_onsets(inp, wlen, ons_idx)
    mx = _np.absolute(_hilbert(segs)).argmax(axis=1) / fs
    mx[mx == 0.0] = 1.0

    return _np.log(mx)


def perceptual_shape(inp: _Array, frqs: _Array) -> tuple:
    """Extracts psychoacoustical features from the spectrum.

    Returns:
        Loudness, roughness, and sharpness.
    """
    if inp.ndim < 2:
        inp = inp[:, None]

    cbrs = _cb.filter_bank(frqs) @ inp
    loud_specific = _np.maximum(_cb.specific_loudness(cbrs), _np.finfo('float64').eps)
    loud_total = array2d_fsum(loud_specific, axis=0)


    z = _np.arange(1, 25)
    sharp = ((z * _cb.weight_factor(z)) @ cbrs) / loud_total
    rough = roughness(inp, frqs)

    return FeatureSpace(loudness=loud_total, sharpness=sharp, roughness=rough)


def loudness(inp: _Array, frqs: _Array) -> _Array:
    """Calculate a measure for the perceived loudness from a spectrogram.

    Args:
        inp:  Magnitude spectrogram.

    Returns:
        Loudness
    """
    cbrs = _cb.filter_bank(frqs) @ inp
    return _cb.total_loudness(cbrs)


def sharpness(inp: _Array, frqs: _Array) -> _Array:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram.

    Args:
        inp:   Two-dimensional input array. Assumed to be an magnitude spectrogram.
        frqs:  Frequency axis of the spectrogram.

    Returns:
        Sharpness for each time instant of the spectragram.
    """
    cbrs = _cb.filter_bank(frqs) @ inp
    return _cb.sharpness(cbrs)


class FeatureSpace:
    """Container class for feature vectors."""
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                val = FeatureSpace(**val)
            self.update(key, val)

    def update(self, key: str, val: Any) -> None:
        self.__dict__[key] = val

    def items(self) -> List[Tuple[str, Any]]:
        """Provides the the FeatureSpace's items.

        Returns:
            List of (key, value) pairs.
        """
        return list(self.__dict__.items())

    def keys(self) -> List[str]:
        """Provides the FeatureSpaces's keys.

        Returns:
            List of keys.
        """
        return list(self.__dict__.keys())

    def values(self) -> List[Any]:
        """Provides the FeatureSpace's values.

        Returns:
            List of values.
        """
        return list(self.__dict__.values())

    def as_dict(self) -> Dict[str, Any]:
        """Returns the FeatureSpace converted to a dict."""
        flat_dict = {}
        for key, val in self.__dict__.items():
            try:
                flat_dict[key] = val.as_dict()
            except AttributeError:
                flat_dict[key] = val
        return flat_dict

    def to_csv(self, path: str = None) -> str:
        """Write FeatureSpace to csv file.

        If ``path`` is ``None``, this method returns of the data of the
        ``FeatureSpace`` as comma seperated values. Otherwise, data is
        written to ``path``.

        Args:
            path:  Output file path.

        Returns:
            FeatureSpace as csv-formatted string if ``path`` is ``None``,
            else ``None``.
        """
        features = {}
        for name, space in self.items():
            try:
                features.update({feat: val for feat, val in space.items()})
            except AttributeError:
                features.update({name: space})

        field_names = ['']
        field_names.extend(features.keys())

        if path is None:
            csv_writer = _csv.DictWriter(_sys.stdout, delimiter=',', fieldnames=field_names)
            self._write(csv_writer, features)
        else:
            with open(path, 'w', newline='') as csv_file:
                csv_writer = _csv.DictWriter(csv_file, delimiter=',', fieldnames=field_names)
                self._write(csv_writer, features)

    @staticmethod
    def _write(csv_writer, features):
        csv_writer.writeheader()

        i = 0
        while True:
            try:
                row = {key: val[i] for key, val in features.items()}
                row[''] = i
                csv_writer.writerow(row)
                i += 1
            except IndexError:
                break

    def to_json(self, path: str = None) -> str:
        """Convert FeaturesSpace to JSON.

        If ``path`` is ``None``, this method returns of the data of the
        ``FeatureSpace`` in JSON format. Otherwise, data is written to
        ``path``.

        Args:
            path:  Output file path.

        Returns:
            FeatureSpace as JSON-formatted string if path is not ``None``,
            else ``None``.
        """
        if path is None:
            return _json.dumps(self.as_dict(), cls=ArrayEncoder)

        with open(path, 'w') as json_file:
            _json.dump(self.as_dict(), json_file, cls=ArrayEncoder)
