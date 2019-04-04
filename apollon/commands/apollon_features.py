import argparse
import sys

import matplotlib.pyplot as plt

from .. audio import load_audio
from .. signal.spectral import stft


def _parse_cml(argv):
    parser = argparse.ArgumentParser(description='Apollon feature extraction engine')

    parser.add_argument('--rhythm', action='store_true',
                        help='Extract features for rhythm track.')

    parser.add_argument('--timbre', action='store_true',
                        help='Extract features for timbre track.')

    parser.add_argument('file_path',  type=str, nargs=1)

    return parser.parse_args(argv)


def _rhythm_track(file_path):
    pass

def _timbre_track(file_path):
    sound = load_audio(file_path)
    spctrgrm = stft(sound.data, sound.fps)
    fig, ax = spctrgrm.plot()
    plt.show()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = _parse_cml(argv)

    if args.rhythm:
        print('Starting rhyhtm track ...')

    if args.timbre:
        print('Starting timbre track ...')
        return _timbre_track(args.file_path[0])

    print('No extractor option specified. Exit.')
    return 123

if __name__ == '__main__':
    sys.exit(main())
