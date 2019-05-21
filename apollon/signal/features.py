# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

import json as _json
import csv as _csv
import sys as _sys

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
        inp  (ndarray)    Input. Each row is assumend FFT bins scaled by `freqs`.
        frqs (ndarray)    One-dimensional array of FFT frequencies.

    Returns:
        (ndarray or scalar)    Spectral centroid frequency.
    """
    inp = _np.atleast_2d(inp).astype('float64')

    weighted_nrgy = _np.multiply(inp, frqs).sum(axis=1).squeeze()
    total_nrgy = inp.sum(axis=1).squeeze()
    total_nrgy[total_nrgy == 0.0] = 1.0

    return _np.divide(weighted_nrgy, total_nrgy)


def spectral_flux(inp: _Array, delta:float=1.0) -> _Array:
    """Estimate the spectral flux

    Args:
        inp   (ndarray)    Input. Each row is assumend FFT bins.
        delta (float)      Sample spacing.

    Returns:
        (ndarray)    Spectral flux.
    """
    inp = _np.atleast_2d(inp).astype('float64')
    return _np.maximum(_np.gradient(inp, delta, axis=-1), 0).squeeze()


def spectral_shape(inp, frqs, low: float = 50, high: float = 16000):
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
        inp  (ndarray)    Input spectrum or spectrogram.
        frqs (ndarray)    Frequency axis.
        low  (float)      Lower cutoff frequency.
        high (float)      Upper cutoff frequency.

    Returns:
        (tuple)    (centroid, spread, skewness, kurtosis)
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
    """Estimate the attack time of each onset an return is logarithm.

    This function estimates the attack time as the duration between the
    onset and the local maxima of the magnitude of the Hilbert transform
    of the local window.

    Args:
        inp     (ndarray)    Input signal.
        fs      (int)        Sampling frequency.
        ons_idx (ndarray)    Sample indices of onsets.
        wlen    (float)      Local window length in samples.

    Returns:
        (ndarray)    Logarithm of the attack time.
    """
    wlen = int(fs * wlen)
    segs = _segment.by_onsets(inp, wlen, ons_idx)
    mx = _np.absolute(_hilbert(segs)).argmax(axis=1) / fs
    mx[mx == 0.0] = 1.0

    return _np.log(mx)


def perceptual_shape(inp: _Array, frqs: _Array) -> tuple:
    """"""
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
        inp (ndarray)    Magnitude spectrogram.

    Returns:
        (ndarray)    Loudness
    """
    cbrs = _cb.filter_bank(frqs) @ inp
    return _cb.total_loudness(cbrs)


def sharpness(inp: _Array, frqs: _Array) -> _Array:
    """Calculate a measure for the perception of auditory sharpness from a spectrogram.

    Args:
        inp  (ndarray)    Two-dimensional input array. Assumed to be an magnitude spectrogram.
        frqs (ndarray)    Frequency axis of the spectrogram.

    Returns:
        (ndarray)    Sharpness for each time instant of the spectragram.
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

    def update(self, key, val):
        self.__dict__[key] = val

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def as_dict(self):
        flat_dict = {}
        for key, val in self.__dict__.items():
            try:
                flat_dict[key] = val.as_dict()
            except AttributeError:
                flat_dict[key] = val
        return flat_dict

    def to_csv(self, path: str = None) -> None:
        """Write FeatureSpace to csv file.

        Args:
            path (str)    Output file path.
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
        """FeaturesSpace in JSON format.

        If ``path`` is None, this method returns the output of json.dump method.
        Otherwise, FeatureSpace is written to ``path``.

        Args:
            path (str)    Output file path.

        Returns:
            (str)     FeatureSpace as JSON string if path is not None
            (None)    If ``path`` is None.
        """
        if path is None:
            return _json.dumps(self.as_dict(), cls=ArrayEncoder)

        with open(path, 'w') as json_file:
            _json.dump(self.as_dict(), json_file, cls=ArrayEncoder)
