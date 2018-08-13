#! python3
# -*- coding: utf-8 -*-

"""hmm_factory/utilities.py

(c) Michael Blaß, 2018

Train PoissonHmm on .wav files.
"""


from optparse import OptionParser
from pathlib import Path
import sys
import typing


def __parse_cml():
    """Insert doc string."""
    usage = 'usage: %prog [OPTIONS] path_to_wav'
    parser = OptionParser(usage=usage)
    parser.add_option('-v', '--verbose', action='store_true',
                      help='enable verbose mode')
    opts, args =  parser.parse_args()

    if len(args) == 0:
        print('No path specified.')
        sys.exit(1)

    return opts, args


def __validate_input_path(str_path):
    """Insert doc string."""
    path = Path(str_path)

    if path.exists():
        if path.is_file():
            yield path.resolve()

        else:
            for p in  path.rglob("*.wav"):
                yield p.resolve()

    else:
        raise FileNotFoundError('Path <{}> could not be found.\n'.format(path))



