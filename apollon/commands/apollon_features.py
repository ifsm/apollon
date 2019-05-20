# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# michael.blass@uni-hamburg.de

import argparse
import json
import sys
import typing

from .. audio import load_audio
from .. io import dump_json, decode_array
from .. signal.spectral import stft
from .. signal.features import FeatureSpace
from .. tools import time_stamp
from .. types import PathType
from .. onsets import FluxOnsetDetector


def _rhythm_track(file_path, out_path):
    snd = load_audio(file_path)

    out = FeatureSpace()
    flux_onsets = FluxOnsetDetector(snd.data, snd.fps)

    spectrogram_params = FeatureSpace(fps=flux_onsets.fps,
                                        n_perseg=flux_onsets.n_perseg,
                                        hop_size=flux_onsets.hop_size,
                                        n_fft=flux_onsets.n_fft,
                                        window=flux_onsets.window,
                                        cutoff=flux_onsets.cutoff)

    features = FeatureSpace(peaks=flux_onsets.peaks,
                            index=flux_onsets.index(),
                            times=flux_onsets.times(snd.fps))

    out.update('meta', {'source': file_path, 'time_stamp': time_stamp()})
    out.update('params', spectrogram_params)
    out.params.update('align', flux_onsets.align)
    out.params.update('peak_picking', flux_onsets.pp_params)
    out.update('features', features)

    if out_path is None:
        out.to_json()
        return 0

    return out.to_json(out_path)


def _timbre_track(file_path: PathType, out_path: PathType = None) -> None:
    """Produce features for the timbre track with standard parameters.

    Args:
        file_path: Path to input file.
        out_path:  Path to output file.
    """
    snd = load_audio(file_path)
    spctrgr = stft(snd.data, snd.fps, n_perseg=1024, hop_size=512)

    features = spctrgr.extract()

    out = {}
    spectrogram_params = ('fps', 'window', 'n_perseg', 'hop_size', 'n_fft')

    out['meta'] = {'source': file_path, 'time_stamp':time_stamp()}
    out['params'] = {param: getattr(spctrgr, param) for param in spectrogram_params}
    out['features'] = features.as_dict()

    dump_json(out, out_path)


def _export_csv(data: typing.Dict[str, typing.Any], path: PathType = None) -> None:
    fspace = json.loads(data, object_hook=decode_array)
    fspace = FeatureSpace(**fspace)
    fspace.to_csv()


def main(args: argparse.Namespace) -> int:
    if args.export:
        if args.export == 'csv':
            _export_csv(args.file[0], args.outpath)
            return 0

    tt_args = (args.file[0], args.outpath)

    if args.rhythm:
        print('Starting rhyhtm track ...')
        return _rhythm_track(*tt_args)

    if args.timbre:
        print('Starting timbre track ...')
        return _timbre_track(*tt_args)

    print('No extractor option specified. Exit.')
    return 123


if __name__ == '__main__':
    sys.exit(main())
