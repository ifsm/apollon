"""
Classes:
    AudioFile   Representation of an audio file.

Functions:
    load_audio   Load .wav file.
"""
import pathlib as _pathlib

import numpy as _np
import matplotlib.pyplot as plt
import soundfile as _sf

from apollon.signal.tools import normalize
from . types import Array as _Array


class AudioFile:
    """Representation of an audio file.
    """
    def __init__(self, path: str, norm: bool = False, mono: bool = True) -> None:
        """Load an audio file.

        Args:
            path:   Path to file.
            norm:   If True, signal will be normalized ]-1, 1[.
            mono:   If True, mixdown all channels.
        """
        self.file = _pathlib.Path(path)
        self.data, self.fps = _sf.read(self.file, dtype='float')
        self.size = self.data.shape[0]

        if mono and self.data.ndim > 1:
            self.data = self.data.sum(axis=1) / self.data.shape[1]

        if norm:
            self.data = normalize(self.data)


    def plot(self):
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(self.data)

    def __str__(self):
        return "<{}, {} kHz, {:.3} s>" \
        .format(self.file.name, self.fps/1000, self.size/self.fps)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.data[item]


def load_audio(path, norm: bool = False, mono: bool = True) -> AudioFile:
    """Load an audio file.

    Params:
        path:   Path to audio file.
        norm:   True if data should be normalized.
        mono:   If True, mixdown channels.

    Return:
        Audio file representation.
    """
    return AudioFile(path, norm)
