# pylint: disable = C0114, C0115,  too-few-public-methods

from dataclasses import dataclass
from pydantic import BaseModel
from apollon.types import NDArray


class SegmentationParams(BaseModel):
    n_perseg: int = 512
    n_overlap: int = 256
    extend: bool = True
    pad: bool = True


class LazySegmentParams(BaseModel):
    n_perseg: int
    n_overlap: int
    norm: bool = False
    mono: bool = True
    expand: bool = True
    dtype: str = "float64"


@dataclass
class Segment:
    idx: int
    start: int
    stop: int
    center: int
    n_frames: int
    data: NDArray
