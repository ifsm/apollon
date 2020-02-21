"""apollon/extractor.py -- Feature extraction framework
"""
import pathlib
from timeit import default_timer as timer
from typing import Tuple, Callable

import numpy as np
import pandas as pd

from . audio import AudioFile
from . segment import Segments, SegmentParams
from . types import PathType
from . types import PathType


@dataclass
class ExtractorFunc:
    name: str
    func: Callable
    kwargs: None = None


class ExtractorBase:
    """Base class for feature extraction pipelines."""
    def __init__(self, verbose: bool = False) -> None:
        """
        Args:
            verbose:  Print status messages during extraction.
        """
        self.verbose = False
        self._params = None
        self._timing = []

    @property
    def params(self):
        """Returns the parameters."""
        return _params

    @property
    def timing(self, total: bool = True) -> str:
        """Returns the total time take for feature extraction."""
        if total:
            return sum(self._timing)
        return self._timing

    def _feedback(self, msg: str) -> None:
        """Print ``msg`` to stdout in verbose mode."""
        if self.verbose:
            print(msg, flush=True)

    def extract(self) -> None:
        """Perform feature extraction."""
        raise NotImplementedError

class AudioExtractor(ExtractorBase):
class IterativeAudioExtractor(ExtractorBase):
    """Extractor pipeline for audio files."""
    def __init__(self, path: PathType, features_names: Tuple[str],
                 segment_params: SegmentParams) -> None:
        """
        Args:
            path:            Path to audio file.
            routines:        Tuple of extraction functions.
            segment_params:  Segmentation parameters.
        """
        super().__init__(routines)
        self._path = pathlib.Path(path)
        self._snd = AudioFile(str(self._path))
        self._feature_names = feature_names
        self._segments = Segments(self._snd, **segment_params.__dict__)
        self._spectrum = Spectrum(spectrum_params)

    @property
    def n_featuress(self) -> int:
        """Return the number of registered estimators."""
        return len(self._feature_names)

    @property
    def path(self) -> str:
        """Returns the path of the audio file."""
        return self._path

    def run(self) -> None:
        """Perform feature extraction."""
        raise NotImplementedError

    def loop(self) -> pd.DataFrame:
        data = np.zeros((self.segments.n_segs, self.n_estimators))
        prev_sxx = Spectrum(self._params.spectrum)
        this_sxx = Spectrum(self._params.sepctrum)
        for seg in self._segments:
            pace = timer()
            sxx.transform(seg.data)
            data[seg.idx] = self.pipeline(seg, sxx, data[seg.idx])
            pace = timer() - pace
            self._timing.append(pace)
            self._feedback(f'Segment {seg.idx}\t Time: {pace:.4} s')

        idx = range(self.segments.n_segs)
        columns = [est.__name__ for est in self._estimators]
        self.features = pd.DataFrame(data, idx, columns)

    def pipeline(self, segment, spectrum):
        """Extraction pipeline."""
        raise NotImplementedError
