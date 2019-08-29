# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

import argparse
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

from .. import io
from .. som.utilities import get_winner


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    weights = io.load(argv.som_file)

    if argv.rt:
        for hmm in argv.objective_files:
            hmm = io.load_json(hmm)
            gamma_ = hmm.params.gamma_.astype('float64').flatten()
            flat_idx = get_winner(weights, gamma_)
            print(np.unravel_index(flat_idx, (30, 30)))

    elif argv.tt:
        fs_path = pathlib.Path(argv.som_file).parent.joinpath('fstats.pkl')
        with fs_path.open('rb') as fobj:
            fstats = pickle.load(fobj)

        feature_mean, feature_std = fstats
        for file_name in argv.objective_files:
            data = io.load_json(file_name)
            feat = data.timbre.features
            timbre_vector = np.array([
                feat.spectral.centroid.mean(), feat.spectral.centroid.std(),
                feat.spectral.spread.mean(), feat.spectral.spread.std(),
                feat.spectral.skewness.mean(), feat.spectral.skewness.std(),
                feat.spectral.kurtosis.mean(), feat.spectral.kurtosis.std(),
                feat.temporal.flux.mean(), feat.temporal.flux.std(),
                feat.perceptual.roughness.mean(), feat.perceptual.roughness.std(),
                feat.perceptual.sharpness.mean(), feat.perceptual.sharpness.std(),
                feat.perceptual.loudness.mean(), feat.perceptual.loudness.std()])

            timbre_vector = (timbre_vector - feature_mean) / feature_std
            flat_idx = get_winner(weights, timbre_vector)
            print(np.unravel_index(flat_idx, (40, 40)))
    else:
        print('Error. Need track info.')
        return -10
    return 0


if __name__ == '__main__':
    sys.exit(main())
