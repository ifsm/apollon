import numpy as np
from pydantic import BaseModel


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
    extend: int | bool = True
    pad: int | bool = True


class Segment:
    idx: int
    start: int
    stop: int
    center: int
    n_frames: int
    data: np.ndarray
