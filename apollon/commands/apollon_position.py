# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

import argparse
import json
import pickle
import sys

import numpy as np
from apollon import io


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv

    with open(argv.som_file, 'rb') as fobj:
        som = pickle.load(fobj)

    for hmm in argv.hmm_files:
        with open(hmm, 'r') as fobj:
            hmm = json.load(fobj, object_hook=io.decode_array)
        gamma_ = hmm['params']['gamma_'].astype('float64').flatten()

        flat_idx, err = som.get_winners(gamma_)
        print(np.unravel_index(flat_idx, som.shape))
    return 0


if __name__ == '__main__':
    sys.exit(main())
