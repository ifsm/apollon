"""apollon/audio.py -- Wrapper classes for audio data.

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß, mblass@posteo.net

Classes:
    AudioFile   Representation of an audio file.

Functions:
    fti16        Cast float to int16.
    load_audio   Load .wav file.
"""
import hashlib
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import soundfile as _sf

from . signal import tools as _ast
from . types import Array, PathType


class AudioFile:
    """Representation of an audio file."""
    def __init__(self, path: PathType) -> None:
        """Load an audio file.

        Args:
            path:   Path to file.
        """
        self._path = pathlib.Path(path)
        self._file = _sf.SoundFile(self.path)

    @property
    def data(self) -> Array:
        """Return audio data as array."""
        return self.read()

    @property
    def file_name(self) -> str:
        """Return source file name."""
        return self._path.name

    @property
    def hash(self) -> str:
        """Compute sha256 hash."""
        obj = hashlib.sha256(self.data.tobytes())
        return obj.hexdigest()

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        return self._file.channels

    @property
    def n_frames(self) -> int:
        """Return number of frames."""
        return self._file.frames

    @property
    def fps(self) -> int:
        """Return sample rate."""
        return self._file.samplerate

    @property
    def path(self) -> str:
        """Return path of audio file."""
        return str(self._path)

    @property
    def shape(self) -> tuple:
        """Return (n_frames, n_channels)."""
        return self.n_frames, self.n_channels

    """
    @property
    def source_id(self) -> SourceId:
        """"""
        return SourceId(self._path.name.split('.')[0], self.hash)
    """
    def close(self) -> None:
        """Close the file."""
        self._file.close()

    def plot(self) -> None:
        """Plot audio as wave form."""
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(self.data)

    def __str__(self):
        return "<{}, {} kHz, {:.3} s>" \
               .format(self._path.name, self.fps/1000, self.n_frames/self.fps)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.n_frames

    def read(self, n_frames: int = None, offset: int = None, norm: bool = False,
             mono: bool = True, dtype: str = 'float64') -> Array:
        # pylint: disable=too-many-arguments
        """Read from audio file.

        Args:
            n_frames:  Number of frames to read.
                       If negative, file is read until EOF.
            offset:    Start position for reading.
            norm:      If ``True``, normalize the data.
            mono:      If ``True``, mixdown all channels.
            dtype:     Dtype of output array.

        Returns:
            Two-dimensional numpy array of shape (n_frames, n_channels).
        """
        n_frames = n_frames or -1
        offset = offset or 0
        if offset >= 0:
            self._file.seek(offset)
            data = self._read(n_frames, dtype=dtype)
        else:
            data = np.zeros((n_frames, self.n_channels))
            n_to_read = offset + n_frames
            if n_to_read > 0:
                self._file.seek(0)
                data[-n_to_read:] = self._read(n_to_read, dtype=dtype)

        if mono and self.n_channels > 1:
            data = data.sum(axis=1, keepdims=True) / self.n_channels
        if norm:
            data = _ast.normalize(data)
        return data

    def _read(self, n_frames: int, dtype: str = 'float64') -> Array:
        return self._file.read(n_frames, dtype=dtype, always_2d=True,
                               fill_value=0)


def fti16(inp: Array) -> Array:
    """Cast audio loaded as float to int16.

    Args:
        inp:    Input array of dtype float64.

    Returns:
        Array of dtype int16.
    """
    return np.clip(np.floor(inp*2**15), -2**15, 2**15-1).astype('int16')


def load_audio(path: PathType) -> AudioFile:
    """Load an audio file.

    Args:
        path:  Path to audio file.

    Return:
        Audio file representation.
    """
    return AudioFile(path)
