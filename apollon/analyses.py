# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

import logging

from . io import dump_json, decode_array
from . signal.spectral import stft, Spectrum
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
    logging.info('Starting rhythm track for {!s}'.format(snd.file))
    onsets = FluxOnsetDetector(snd.data, snd.fps)
    segs = segment.by_onsets(snd.data, 2**11, onsets.index())
    spctr = Spectrum(segs, snd.fps, window='hamming', n_fft=2**15)

    onsets_features = {
        'peaks': onsets.peaks,
        'index': onsets.index(),
        'times': onsets.times(snd.fps)
    }

    track_data = {
        'meta': {'source': str(snd.file.absolute()),
                 'time_stamp': time_stamp()},
        'params': {'onsets': onsets.params(), 'spectrum': spctr.params()},
        'features': {'onsets': onsets_features,
                     'spectrum':
                      spctr.extract(cf_low=100, cf_high=9000).as_dict()}
    }
    logging.info('Done with rhythm track for {!s}.'.format(snd.file))
    return track_data


def timbre_track(snd: AudioFile) -> dict:
    """Perform timbre track analysis of given audio file.

    The timbre track extracts audio features related the perception
    of timbre with  high frequency resolution of 1.35 Hz given 44,1 kHz
    sampling frequency.

    the FFT is set to 
    Args:
        snd:  Sound data.
        fps:  Sample rate.

    Returns:
        Timbre track parameters and data.
    """
    logging.info('Starting timbre track for {!s}.'.format(snd.file))
    spctrgr = stft(snd.data, snd.fps, n_perseg=2**15, hop_size=2**14)

    track_data = {
        'meta': {'source': str(snd.file.absolute()),
                 'time_stamp':time_stamp()},
        'params': {'spectrogram': spctrgr.params()},
        'features': spctrgr.extract(cf_low=50, cf_high=15000).as_dict()
    }
    print(spctrgr.params())
    logging.info('Done with timbre track for {!s}.'.format(snd.file))
    return track_data
