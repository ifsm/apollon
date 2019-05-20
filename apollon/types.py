"""
types.py -- Apollon type hints.
Copyright (C) 2018  Michael Blaß <michael.blass@uni-hamburg.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import pathlib
from typing import (Any, Dict, Generator, Iterable, List, Optional, Tuple, Union)
import numpy as _np    # type: ignore

Array = _np.ndarray    # pylint: disable = C0103

ArrayOrStr = Union[Array, str]
IterOrNone = Union[Iterable, None]

ParamsType = Dict[str, Any]
ParameterSet = Optional[ParamsType]
ParserType = Tuple[ParamsType, List[str]]
PathType = Union[str, pathlib.Path]
PathGen = Generator[PathType, None, None]
