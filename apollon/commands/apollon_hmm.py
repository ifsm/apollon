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


def _load_track_file(track_file: str) -> dict:
    track_file = pathlib.Path(track_file)
    with track_file.open('r') as fobj:
        track_data = json.load(fobj, object_hook=io.decode_array)
    return track_data


def _parse_feature(track_data: dict, feature_path: str) -> _Array:
    feature = track_data
    for key in feature_path.split('.'):
        try:
            feature = feature[key]
        except KeyError:
            print('Error. Invalid node "{}" in feature path.'.format(key))
            exit(10)
    return feature


def _generate_outpath(in_path, out_path: str, feature_path: str) -> None:
    in_path = pathlib.Path(in_path)
    default_fname = '{}.hmm'.format(in_path.stem)
    if out_path is None:
        out_path = pathlib.Path(default_fname)
    else:
        out_path = pathlib.Path(out_path)
        if not out_path.suffix:
            out_path = out_path.joinpath(default_fname)
        if not out_path.parent.is_dir():
            print('Error. Path "{!s}" does not exist.'.format(out_path.parent))
            exit(10)
    return out_path


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
        exit(10)

    return min(trails, key=lambda hmm: abs(hmm.quality.nll))


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    for trf in argv.track_files:
        track_data = _load_track_file(trf)
        feature = _parse_feature(track_data, argv.feature_path)
        hmm = _train_n_hmm(feature, argv.mstates, 5)
        out_path = _generate_outpath(trf, argv.outpath, argv.feature_path)
        io.dump_json(hmm.to_dict(), out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
