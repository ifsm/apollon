# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de


import argparse
import itertools
import json
import multiprocessing
import sys
import typing
import soundfile

from .. import analyses
from .. types import PathType
from .. import io
from .. signal.features import FeatureSpace
from .. audio import load_audio

class ShortPiece(Exception):
    pass

class BadSampleRate(Exception):
    pass


def _check_audio(path):
    snd_info = sf.info(path)
    if snd_info.duration < 30:
        raise ShortPiece('Duration of {} is less than {} s.'.format(path, 30))

    if snd_info.samplerate != 44100:
        raise BadSampleRate('Sample rate of {} Hz cannot be processed'.format(snd_info.samplerate))


def main(argv: dict = None) -> int:
    if argv is None:
        argv = sys.argv

    args = itertools.product(argv.files, [argv])
    with multiprocessing.Pool(processes=12) as pool:
        pool.starmap(_feature_extraction, args)
    return 0


def _feature_extraction(path, args) -> None:
    _check_audio(snd)
    snd = load_audio(path)
    snd.cut(snd.fps*2, snd.size-(snd.fps*5))
    snd_cut = snd[snd.fps*2:-snd.fps*5]

    track_data = {}
    if args.rhythm:
        track_data['rhythm'] = analyses.rhythm_track(snd_cut, snd.fps)

    if args.timbre:
        track_data['timbre'] = analyses.timbre_track(snd_cut, snd.fps)

    if args.pitch:
        track_data['pitch'] = analyses.pitch_track(snd_cut, snd.fps)

    out_path = io.generate_outpath(path, args.outpath, 'feat')
    io.dump_json(track_data, out_path)


if __name__ == '__main__':
    sys.exit(main())
