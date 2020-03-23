# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net


import argparse
import itertools
import json
import sys
import typing
import soundfile as sf
import logging
import time
import warnings

from .. import analyses
from .. types import PathType
from .. import io
from .. audio import load_audio


def _check_audio(path):
    snd_info = sf.info(path)
    if snd_info.duration < 30:
        warnings.warn('Duration of {} is less than {} s.'.format(path, 30))

    if snd_info.samplerate != 44100:
        warnings.warn('Sample rate of {} Hz cannot be processed'.format(snd_info.samplerate))


def main(argv: dict = None) -> int:
    logging.basicConfig(filename='fe.log', filemode='w', level=logging.DEBUG)
    logging.captureWarnings(True)
    if argv is None:
        argv = sys.argv

    timer_total = time.time()
    for path in argv.files:
        logging.info('Loading {}'.format(path))
        print('Processing: {}'.format(path), end=' ... ', flush=True)
        timer_start = time.time()

        _check_audio(path)
        snd = load_audio(path)
        snd.cut(snd.fps*2, snd.size-(snd.fps*5))

        track_data = {}
        if argv.rhythm:
            track_data['rhythm'] = analyses.rhythm_track(snd)

        if argv.timbre:
            track_data['timbre'] = analyses.timbre_track(snd)

        if argv.pitch:
            track_data['pitch'] = analyses.pitch_track(snd)

        out_path = io.generate_outpath(path, argv.outpath, 'feat')
        io.dump_json(track_data, out_path)
        timer_stop = time.time()
        print('Done in {:.5} s.'.format(timer_stop-timer_start), flush=True)

    logging.info('--- JOB DONE ---')
    print('Job done. Total time: {:.5} s.'.format(time.time()-timer_total))
    return 0


if __name__ == '__main__':
    sys.exit(main())
