# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de


import argparse
import json
import sys
import typing

from .. import analyses
from .. types import PathType
from .. import io
from .. signal.features import FeatureSpace


def main(argv: dict = None) -> int:
    if argv is None:
        argv = sys.argv

    for path in argv.files:
        track_data = {}
        if argv.rhythm:
            track_data['rhythm'] = analyses.rhythm_track(path)

        if argv.timbre:
            track_data['timbre'] = analyses.timbre_track(path)

        if argv.pitch:
            track_data['pitch'] = analyses.pitch_track(path)

        out_path = io.generate_outpath(path, argv.outpath, 'feat')
        io.dump_json(track_data, out_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
