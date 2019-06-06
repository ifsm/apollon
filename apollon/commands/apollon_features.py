# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de


import argparse
import itertools
import json
import multiprocessing
import sys
import typing

from .. import analyses
from .. types import PathType
from .. import io
from .. signal.features import FeatureSpace


def main(argv: dict = None) -> int:
    if argv is None:
        argv = sys.argv

    args = itertools.product(argv.files, [argv])
    with multiprocessing.Pool(processes=24) as pool:
        pool.starmap(_feature_extraction, args)

    return 0


def _feature_extraction(path, args) -> None:
    track_data = {}
    if args.rhythm:
        track_data['rhythm'] = analyses.rhythm_track(path)

    if args.timbre:
        track_data['timbre'] = analyses.timbre_track(path)

    if args.pitch:
        track_data['pitch'] = analyses.pitch_track(path)

    out_path = io.generate_outpath(path, args.outpath, 'feat')
    io.dump_json(track_data, out_path)


if __name__ == '__main__':
    sys.exit(main())
