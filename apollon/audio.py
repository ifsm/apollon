"""apollon/audio.py -- Wrapper classes for audio data.

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net

Classes:
    AudioFile   Representation of an audio file.

Functions:
    fti16        Cast float to int16.
    load_audio   Load .wav file.
"""
from dataclasses import dataclass
import pathlib as _pathlib
from typing import Generator

import matplotlib.pyplot as _plt
import numpy as np
import soundfile as _sf

from . signal import tools as _ast
from . types import Array, PathType


class AudioFile:
    """Representation of an audio file.

        Args:
            path:   Path to file.
            norm:   If True, signal will be normalized ]-1, 1[.
            mono:   If True, mixdown all channels.
    """
    def __init__(self, path: PathType, norm: bool = False,
                 mono: bool = True) -> None:
        """Load an audio file."""

        self._path = _pathlib.Path(path)
        self._file = _sf.SoundFile(self.path)
        self.norm = norm
        self.mono = mono

    @property
    def data(self) -> np.ndarray:
        data = self._file.read(always_2d=True)
        if self.mono and self.n_channels > 1:
            data = data.sum(axis=1, keepdims=True) / self.n_channels
        if self.norm:
            data = _ast.normalize(data)
        self._file.seek(0)
        return data

    @property
    def n_channels(self) -> int:
        return self._file.channels

    @property
    def n_frames(self) -> int:
        return self._file.frames

    @property
    def fps(self) -> int:
        return self._file.samplerate

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def shape(self) -> tuple:
        return self.n_frames, self.n_channels

    def close(self) -> None:
        self._file.close()

    def plot(self) -> None:
        """Plot audio as wave form."""
        fig = _plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(self.data)

    def __str__(self):
        return "<{}, {} kHz, {:.3} s>" \
               .format(self._path.name, self.fps/1000, self.n_frames/self.fps)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.size


def fti16(inp: Array) -> Array:
    """Cast audio loaded as float to int16.

    Params:
        inp:    Input array of dtype float64.

    Returns:
        Array of dtype int16.
    """
    return np.clip(np.floor(inp*2**15), -2**15, 2**15-1).astype('int16')


def load_audio(path: PathType, norm: bool = False, mono: bool = True
          ) -> AudioFile:
    """Load an audio file.

    Args:
        path:   Path to audio file.
        norm:   True if data should be normalized.
        mono:   If True, mixdown channels.

    Return:
        Audio file representation.
    """
    return AudioFile(path, norm, mono)
