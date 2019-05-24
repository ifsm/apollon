#!/usr/bin/env python3

# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de


import argparse
import json
import pathlib
import sys
import typing

from .. import io
from .. hmm import PoissonHmm
from .. types import Array as _Array

def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    track_file = pathlib.Path(argv.track_file)
    if track_file.exists() and track_file.is_file():
        with track_file.open('r') as fobj:
            track_data = json.load(fobj, object_hook=io.decode_array)

    feature = track_data
    for key in argv.feature_path.split('.'):
        try:
            feature = feature[key]
        except KeyError:
            print('Error. Invalid node "{}" in feature path.'.format(key))
            return 10

    hmm = _train_n_hmm(feature, 3, 5)
    if hmm is None:
        return 10

    default_fname = argv.feature_path.replace('.', '_') + '.hmm'
    if argv.outpath is None:
        out_path = pathlib.Path(default_fname)
    else:
        out_path = pathlib.Path(argv.outpath)
        if not out_path.suffix:
            out_path = out_path.joinpath(default_fname)
        if not out_path.parent.is_dir():
            print('Error. Path "{!s}" does not exist.'.format(out_path.parent))
            return 10
    print(out_path)

    #io.dump_json(hmm.to_dict(), out_path)
    return 0


def _train_n_hmm(data: _Array, m_states: int, n_trails: int):
    """Trains ``n_trails`` HMMs each initialized with a random tpm.

    Args:
        data:      Possibly unporcessed input data set.
        m_states:  Number of states.
        n_trails:  Number of trails.

    Returns:
        Best model regarding to log-likelihood.
    """
    feat = data.round().astype(int)
    trails = []
    for i in range(n_trails):
        hmm = PoissonHmm(feat, m_states, init_gamma='softmax')
        hmm.fit(feat)
        if hmm.success:
            trails.append(hmm)

    if len(trails) == 0:
        print('Error. Could not train HMM.')
        return None

    return min(trails, key=lambda hmm: abs(hmm.quality.nll))


if __name__ == '__main__':
    sys.exit(main())
