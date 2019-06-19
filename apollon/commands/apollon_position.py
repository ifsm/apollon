# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

import argparse
import pathlib
import pickle
import sys
import pandas as pd
import numpy as np
from .. import io
from .. tools import standardize
from .. som.utilities import get_winner

def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv
    
    weights = io.load(argv.som_file)

    if argv.rt:
        for hmm in argv.objective_files:
            hmm = io.load_json(hmm)
            gamma_ = hmm.params.gamma_.astype('float64').flatten()
            flat_idx= get_winner(weights, gamma_)
            print(np.unravel_index(flat_idx, (30, 30)))

    elif argv.tt:
        sfm_path = pathlib.Path(argv.som_file).parent
        sfm_path = sfm_path.joinpath('timbre_track.sfm')
        sfm = pd.read_csv(sfm_path, index_col=0)
        sfm = sfm.values

        for feat in argv.objective_files:
            feat = io.load_json(feat)
            feat = feat.timbre.features
            timbre_vector = np.array([
                feat.spectral.centroid.sum(),
                feat.spectral.spread.sum(),
                feat.spectral.skewness.sum(),
                feat.spectral.kurtosis.sum(),
                feat.temporal.flux.sum(),
                feat.perceptual.roughness.sum(),
                feat.perceptual.sharpness.sum(),
                feat.perceptual.loudness.sum()])
            
            updated_sfm = np.vstack((sfm, timbre_vector))
            timbre_vector = standardize(updated_sfm)[-1]
            flat_idx = get_winner(weights, timbre_vector)
            print(np.unravel_index(flat_idx, (25, 25)))
    else:
        print('Error. Need track info.')
        return -10
    return 0


if __name__ == '__main__':
    sys.exit(main())
