"""
Type aliases
"""

import pathlib
from typing import (Any, Collection, Dict, Generator, Iterable, List, Optional,
                    Sequence, Tuple, Union)
import numpy as np
import numpy.typing as npt
from matplotlib import axes


Array = np.ndarray
IntArray = np.ndarray[Any, np.dtype[np.int_]]
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

ArrayOrStr = Union[Array, str]
IterOrNone = Union[Iterable, None]

ParamsType = Dict[str, Any]
PathType = Union[str, pathlib.Path]
PathGen = Generator[PathType, None, None]
Schema = Dict[str, Collection[str]]

Shape = Tuple[int, int]
SomDims = Tuple[int, int, int]
Coord = Tuple[int, int]
AdIndex = Tuple[List[int], List[int]]

Axis = axes._axes.Axes

def floatarray(inp: Any) -> FloatArray:
    return np.asanyarray(inp, dtype=np.float64)
