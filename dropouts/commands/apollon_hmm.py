#!/usr/bin/env python3

# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net


import argparse
import json
import pathlib
import sys
import typing

import numpy as np

from .. import io
from .. hmm import PoissonHmm
from .. types import Array as _Array
from .. import tools


def _load_track_file(track_file: str) -> dict:
    track_file = pathlib.Path(track_file)
    with track_file.open('r') as fobj:
        track_data = json.load(fobj, object_hook=io.decode_array)
    return track_data


def _parse_feature_file(track_file: str, feature_path: str) -> _Array:
    feature = _load_track_file(track_file)
    for key in feature_path.split('.'):
        try:
            feature = feature[key]
        except KeyError:
            print('Error. Node "{}" not in "{}".'.format(key, track_file))
            exit(10)
    return _scaling(feature, key)



def _scaling(data: _Array, feature: str) -> _Array:
    features1000 = ['skewness', 'kurtosis', 'loudness',
                    'roughness', 'sharpness']
    if feature in features1000:
        out = tools.scale(data, 1, 1000)
    else:
        out = data
    return out.round().astype(int)


def _train_n_hmm(data: _Array, m_states: int, n_trails: int):
    """Trains ``n_trails`` HMMs each initialized with a random tpm.

    Args:
        data:      Possibly unporcessed input data set.
        m_states:  Number of states.
        n_trails:  Number of trails.

    Returns:
        Best model regarding to log-likelihood.
    """
    trails = []
    for i in range(n_trails):
        hmm = PoissonHmm(data, m_states, init_lambda='hist', init_gamma='softmax')
        hmm.fit(data)
        if hmm.success and not np.isnan(hmm.quality.nll):
            trails.append(hmm)

    for i in range(n_trails):
        hmm = PoissonHmm(data, m_states, init_lambda='linear', init_gamma='softmax')
        hmm.fit(data)
        if hmm.success and not np.isnan(hmm.quality.nll):
            trails.append(hmm)

    for i in range(n_trails):
        hmm = PoissonHmm(data, m_states, init_lambda='quantile', init_gamma='softmax')
        hmm.fit(data)
        if hmm.success and not np.isnan(hmm.quality.nll):
            trails.append(hmm)

    if len(trails) == 0:
        return None
    return min(trails, key=lambda hmm: abs(hmm.quality.nll))


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    for trf in argv.track_files:
        feature = _parse_feature_file(trf, argv.feature_path)
        hmm = _train_n_hmm(feature, argv.mstates, 5)
        if hmm is None:
            print('Error. Could not train HMM on {}'.format(trf))
            continue
        out_path = io.generate_outpath(trf, argv.outpath, 'hmm')
        io.dump_json(hmm.to_dict(), out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
