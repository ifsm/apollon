# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de


from . audio import load_audio
from . io import dump_json, decode_array
from . signal.spectral import stft, Spectrum
from . signal.features import FeatureSpace
from . tools import time_stamp
from . types import PathType
from . types import Array as _Array
from . onsets import FluxOnsetDetector
from . import segment

def rhythm_track(file_path: PathType) -> dict:
    """Perform rhythm track analysis of given audio file.

    Args:
        file_path:  Path to audio file.

    Returns:
        Rhythm track parameters and data.
    """
    snd = load_audio(file_path)
    onsets = FluxOnsetDetector(snd.data, snd.fps)
    segs = segment.by_onsets(snd.data, 2**11, onsets.index())
    spctr = Spectrum(segs, snd.fps, window='hamming')

    onsets_features = {
        'peaks': onsets.peaks,
        'index': onsets.index(),
        'times': onsets.times(snd.fps)
    }

    track_data = {
        'meta': {'source': file_path, 'time_stamp': time_stamp()},
        'params': {'onsets': onsets.params(), 'spectrum': spctr.params()},
        'features': {'onsets': onsets_features,
                     'spectrum': spctr.extract().as_dict()}
    }


    return track_data


def timbre_track(file_path: PathType) -> dict:
    """Perform timbre track analysis of given audio file.

    Args:
        file_path:  Path to input file.

    Returns:
        Timbre track parameters and data.
    """
    snd = load_audio(file_path)
    spctrgr = stft(snd.data, snd.fps, n_perseg=1024, hop_size=512)

    track_data = {
        'meta': {'source': file_path, 'time_stamp':time_stamp()},
        'params': {'spectrogram': spctrgr.params()},
        'features': spctrgr.extract().as_dict()
    }
    return track_data
