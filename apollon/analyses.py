# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de


from . io import dump_json, decode_array
from . signal.spectral import stft, Spectrum
from . signal.features import FeatureSpace
from . tools import time_stamp
from . types import PathType
from . types import Array as _Array
from . onsets import FluxOnsetDetector
from . import segment
from . audio import AudioFile


def rhythm_track(snd: AudioFile) -> dict:
    """Perform rhythm track analysis of given audio file.

    Args:
        snd:  Sound data.
        fps:  Sample rate.

    Returns:
        Rhythm track parameters and data.

    Raises:
        ShortPiece
    """

    onsets = FluxOnsetDetector(snd, fps)
    segs = segment.by_onsets(snd, 2**11, onsets.index())
    spctr = Spectrum(segs, fps, window='hamming')

    onsets_features = {
        'peaks': onsets.peaks,
        'index': onsets.index(),
        'times': onsets.times(fps)
    }

    track_data = {
        'meta': {'source': snd.file.absolute(), 'time_stamp': time_stamp()},
        'params': {'onsets': onsets.params(), 'spectrum': spctr.params()},
        'features': {'onsets': onsets_features,
                     'spectrum': spctr.extract(cf_low=100, cf_high=9000).as_dict()}
    }
    return track_data


def timbre_track(snd: AudioFile) -> dict:
    """Perform timbre track analysis of given audio file.

    Args:
        snd:  Sound data.
        fps:  Sample rate.

    Returns:
        Timbre track parameters and data.
    """
    spctrgr = stft(snd_cut, fps, n_perseg=2048, hop_size=204)

    track_data = {
        'meta': {'source': file_path, 'time_stamp':time_stamp()},
        'params': {'spectrogram': spctrgr.params()},
        'features': spctrgr.extract(cf_low=50, cf_high=15000).as_dict()
    }
    return track_data
