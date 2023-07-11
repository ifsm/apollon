"""
Type aliases
"""

import pathlib
from typing import (Any, Collection, Dict, Generator, Iterable, List,
                    Tuple)
import numpy as np


Array = np.ndarray
NDArray = np.ndarray[Any, Any]
IntArray = np.ndarray[Any, np.dtype[np.int_]]
Int16Array = np.ndarray[Any, np.dtype[np.int16]]
FloatArray = np.ndarray[Any, np.dtype[np.double]]
ComplexArray = np.ndarray[Any, np.dtype[np.complex128]]

ArrayOrStr = Array | str
IterOrNone = Iterable | None

ParamsType = Dict[str, Any]
PathType = pathlib.Path | str
PathGen = Generator[PathType, None, None]
Schema = Dict[str, Collection[str]]

Shape = Tuple[int, int]
SomDims = Tuple[int, int, int]
Coord = Tuple[int, int]
AdIndex = Tuple[List[int], List[int]]

def floatarray(inp: Any) -> FloatArray:
    """Cast sequence type to float64 array"""
    return np.asanyarray(inp, dtype=np.double)
