"""
Classes:
    _AudioData

Functions:
    loadwav         Load .wav file.
"""


import numpy as _np
import soundfile as _sf

from apollon.io import WavFileAccessControl as _WavFileAccessControl
from apollon.signal.tools import normalize


class _AudioData:

    __slots__ = ['_fs', '_data']

    # Descriptor attribute
    file = _WavFileAccessControl()

    def __init__(self, file_name, norm=True):
        """Representation of an audio file.

        Params:
            file_name   (str)   Name of file.
            norm        (bool)  If True, signal will be normalized.

        Return:
            (AudioData) Object
        """
        self.file = file_name

        self._data, self._fs = _sf.read(file_name, dtype='float64')

        if self._data.ndim == 2:
            self._data = self._data.sum(axis=1) / 2

        if norm:
            self._data = normalize(self._data)

    @property
    def fs(self) -> int:
        """Return sample rate."""
        return self._fs

    @property
    def data(self, n: int = None) -> _np.ndarray:
        """Return the audio frames as ints.

        Args:
            n: (int)    Return only the first n frames (default = None)

        Returns:
            (np.ndarray) frames
        """
        return self._data[:n]


    def plot(self, tickunit='seconds'):
        _aplot.signal(self, xaxis=tickunit)

    def __str__(self):
        return "<{}, fs: {}, N: {}>" \
        .format(self.file.name, self.fs, len(self))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self._data.size

    def __getitem__(self, item):
        return self._data[item]


def loadwav(path, norm=True):
    """Load a .wav file.

    Params:
        path    (str or fobject)
        norm    (bool) True if data should be normalized.

    Return:
        (_AudioData) object.
    """
    return _AudioData(path, norm)
