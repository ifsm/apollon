import argparse
import sys

from .. audio import load_audio
from .. io import dump_json
from .. signal.spectral import stft
from .. tools import time_stamp
from .. types import PathType


def _parse_cml(argv):
    parser = argparse.ArgumentParser(description='Apollon feature extraction engine')

    parser.add_argument('--rhythm', action='store_true',
                        help='Extract features for rhythm track.')

    parser.add_argument('--timbre', action='store_true',
                        help='Extract features for timbre track.')

    parser.add_argument('-o', '--outpath', action='store',
                        help='Output file path.')

    parser.add_argument('filepath', type=str, nargs=1)
    return parser.parse_args(argv)


def _rhythm_track(file_path):
    pass


def _timbre_track(file_path: PathType, out_path: PathType = None) -> None:
    """Produce features for the timbre track with standard parameters.

    Args:
        file_path (PathType)    Path to input file.
        out_path  (PathType)    Path to output file.

    Returns:
        None    If ``out_path`` is not None
        (str)   JSON encoded features, otherwise.
    """
    snd = load_audio(file_path)
    spctrgr = stft(snd.data, snd.fps, n_perseg=1024, hop_size=512)

    features = spctrgr.extract()

    out = {}
    spectrogram_params = ('fps', 'window', 'n_perseg', 'hop_size', 'n_fft')

    out['meta'] = {'source': file_path, 'time_stamp':time_stamp()}
    out['params'] = {param: getattr(spctrgr, param) for param in spectrogram_params}
    out['features'] = features.as_dict()

    if out_path is not None:
        dump_json(out, out_path)
        return 0
    return dump_json(out)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = _parse_cml(argv)

    if args.rhythm:
        print('Starting rhyhtm track ...')

    if args.timbre:
        print('Starting timbre track ...')
        _args = (args.filepath[0], args.outpath)
        return _timbre_track(*_args)

    print('No extractor option specified. Exit.')
    return 123

if __name__ == '__main__':
    sys.exit(main())
