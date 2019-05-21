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

def _export_csv(
        data: typing.Dict[str, typing.Any],
        path: PathType = None) -> None:
    """"""
    fspace = json.loads(data, object_hook=decode_array)
    fspace = FeatureSpace(**fspace)
    fspace.to_csv()


def main(args: argparse.Namespace) -> int:
    if args.export:
        if args.export == 'csv':
            _export_csv(args.file[0], args.outpath)
            return 0

    track_data = {}
    if args.rhythm:
        track_data['rhythm'] = analyses.rhythm_track(args.file[0])

    if args.timbre:
        track_data['timbre'] = analyses.timbre_track(args.file[0])

    if args.pitch:
        track_data['pitch'] = analyses.pitch_track(args.file[0])

    io.dump_json(track_data, args.outpath)

    return 0


if __name__ == '__main__':
    sys.exit(main())
