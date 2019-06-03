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


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    som_path = pathlib.Path(argv.som_path)
    hmm_path = pathlib.Path(argv.hmm_path)

    with som_path.open('r') as fobj:
        som = json.load(fobj, object_hook=io.decode_array)

    with hmm_path.open('r') as fobj:
        hmm = json.load(fobj, object_hool=io.decode_array)

    return 0


if __name__ == '__main__':
    sys.exit(main())
