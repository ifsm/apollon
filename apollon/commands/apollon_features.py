# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de


import argparse
import itertools
import json
import multiprocessing
import sys
import typing
import soundfile as sf
import logging

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
        logging.error('Piece to short: {}'.format(path))
        # raise ShortPiece('Duration of {} is less than {} s.'.format(path, 30))
        return 10

    if snd_info.samplerate != 44100:
        logging.error('Bad sample rate: {}'.format(path))
        # raise BadSampleRate('Sample rate of {} Hz cannot be processed'.format(snd_info.samplerate))
        return 10
    return 0

def main(argv: dict = None) -> int:
    logging.basicConfig(filename='fe.log', filemode='w', level=logging.DEBUG)
    if argv is None:
        argv = sys.argv

    args = itertools.product(argv.files, [argv])
    n_processes = 3 
    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.starmap(_feature_extraction, args)
    return 0


def _feature_extraction(path, args) -> None:
    logging.info('Loading {}'.format(path))

    if _check_audio(path) != 0:
        return 10

    snd = load_audio(path)
    snd.cut(snd.fps*2, snd.size-(snd.fps*5))

    track_data = {}
    if args.rhythm:
        track_data['rhythm'] = analyses.rhythm_track(snd)

    if args.timbre:
        track_data['timbre'] = analyses.timbre_track(snd)

    if args.pitch:
        track_data['pitch'] = analyses.pitch_track(snd_cut, snd.fps)

    out_path = io.generate_outpath(path, args.outpath, 'feat')
    io.dump_json(track_data, out_path)


if __name__ == '__main__':
    sys.exit(main())
