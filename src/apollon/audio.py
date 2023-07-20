"""
Audio file representation
"""
import hashlib
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import soundfile as _sf

from . signal import tools as _ast
from . types import FloatArray, NDArray, PathType


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
    def data(self) -> FloatArray:
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
        return int(self._file.channels)

    @property
    def n_frames(self) -> int:
        """Return number of frames."""
        return int(self._file.frames)

    @property
    def fps(self) -> int:
        """Return sample rate."""
        return int(self._file.samplerate)

    @property
    def path(self) -> str:
        """Return path of audio file."""
        return str(self._path)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_frames, n_channels)."""
        return self.n_frames, self.n_channels

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    def plot(self) -> None:
        """Plot audio as wave form."""
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(self.data)

    def __str__(self) -> str:
        return (f"<{self._path.name}, {self.fps/1000} kHz, "
                f"{self.n_frames/self.fps:.3} s>")

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.n_frames

    def read(self, n_frames: int | None = None, offset: int | None = None,
             norm: bool = False, mono: bool = True, dtype: str = 'float64'
        ) -> NDArray:
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

    def _read(self, n_frames: int, dtype: str = 'float64') -> NDArray:
        ffs = self.n_frames if n_frames == -1 else n_frames
        out = np.empty((ffs, self.n_channels), dtype=dtype)
        self._file.read(n_frames, dtype=dtype, always_2d=True, fill_value=0,
                        out=out)
        return out


def load_audio(path: PathType) -> AudioFile:
    """Load an audio file.

    Args:
        path:  Path to audio file.

    Return:
        Audio file representation.
    """
    return AudioFile(path)
