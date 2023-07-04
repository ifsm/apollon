# pylint: disable = missing-class-docstring,too-few-public-methods
"""
common models
"""

from dataclasses import dataclass

from pydantic import BaseModel

from . signal.models import StftParams
from . types import NDArray


class PeakPickingParams(BaseModel):
    n_before: int
    n_after: int
    alpha: float
    delta: float


class LazySegmentParams(BaseModel):
    n_perseg: int
    n_overlap: int
    norm: bool = False
    mono: bool = True
    expand: bool = True
    dtype: str = "float64"


class SegmentationParams(BaseModel):
    n_perseg: int = 512
    n_overlap: int = 256
    extend: bool = True
    pad: bool = True


@dataclass
class Segment:
    idx: int
    start: int
    stop: int
    center: int
    n_frames: int
    data: NDArray




class FluxOnsetDetectorParams(BaseModel):
    stft_params: StftParams
    pp_params: PeakPickingParams
