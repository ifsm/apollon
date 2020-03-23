# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

import argparse
import multiprocessing as mp
import sys

from .. import onsets


def _parse_cml(argv):
    parser = argparse.ArgumentParser(description='Apollon onset detection engine')

    parser.add_argument('--amplitude', action='store_true',
                        help='Detect onsets based on local extrema in the time domain signal.')

    parser.add_argument('--entropy', action='store_true',
                        help='Detect onsets based on time domain entropy maxima.')

    parser.add_argument('--flux', action='store_true',
                        help='Detect onsets based on spectral flux.')

    parser.add_argument('-o', '--outpath', action='store',
                        help='Output file path.')

    parser.add_argument('filepath', type=str, nargs=1)
    return parser.parse_args(argv)


def _amp(a):
    print('Amplitude')
    return a

def _entropy(a):
    print('Entropy')
    return a

def _flux(a):
    print('Flux')
    return a


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = _parse_cml(argv)


    args = _parse_cml(argv)
    detectors = {'amplitude': _amp,
                 'entropy': _entropy,
                 'flux': _flux}

    methods = [func for name, func in detectors.items() if getattr(args, name)]
    if len(methods) == 0:
        print('At least one detection method required. Aborting.')
        return 1

    with mp.Pool(processes=3) as pool:
        results = [pool.apply_async(meth, (i,)) for i, meth in enumerate(methods)]
        out = [res.get() for res in results]
    return out

if __name__ == '__main__':
    sys.exit(main())
