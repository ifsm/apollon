# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

import argparse
import json
import sys
import typing

from .. io import dump_json, decode_array
from .. signal.spectral import stft
from .. signal.features import FeatureSpace
from .. tools import time_stamp
from .. types import PathType


def _parse_cml(argv):
    parser = argparse.ArgumentParser(description='Apollon feature extraction engine')

    parser.add_argument('--csv', action='store_true',
                        help='Export csv')

    parser.add_argument('-o', '--outpath', action='store',
                        help='Output file path')

    parser.add_argument('csv_data', type=str, nargs=1)

    return parser.parse_args(argv)


def _export_csv(data: typing.Dict[str, typing.Any], path: PathType = None) -> None:

    fspace = json.loads(data, object_hook=decode_array)
    fspace = FeatureSpace(**fspace)
    fspace.to_csv()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = _parse_cml(argv)

    if args.csv:
        _export_csv(args.csv_data[0], args.outpath)
        return 0

if __name__ == '__main__':
    sys.exit(main)
