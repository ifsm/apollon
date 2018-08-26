#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""get_som_winner.py

(c) Michael Blaß, 2016

Return index of winning neuron given a SOM and a HMM.
"""


import sys
from optparse import OptionParser
import pathlib

from numpy import atleast_2d
from numpy import unravel_index

from apollon import io


def main():

    usage = 'usage: %prog [OPTIONS] som_file hmm_file'
    parser = OptionParser(usage=usage)
    parser.add_option('-f', '--flat', action='store_true',
                      help='return flat index')

    (opts, args) = parser.parse_args()
    if len(args) != 2:
        print('Specify exactly two arguments (som_file, hmm_file).')
        sys.exit(1)

    som_file = pathlib.Path(args[0])
    hmm_file = pathlib.Path(args[1])

    if som_file.exists() and str(som_file).endswith('.som'):
        if hmm_file.exists() and str(hmm_file).endswith('.hmm'):
            som = io.load(str(som_file))
            hmm = io.load(str(hmm_file))
        else:
            raise FileNotFoundError('File {} not found or is no valid HMM.'
                                    .format(args[1]))
    else:
        raise FileNotFoundError('File {} not found or is no valid SOM.'
                                .format(args[0]))

    foo = som.get_winners(hmm.reshape(16))[0]
    if opts.flat:
        print(foo)
    else:
        x, y = unravel_index(foo, som.shape[:2])
        print("{},{}".format(x, y))

if __name__ == "__main__":
    sys.exit(main())

