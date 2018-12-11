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

import pathlib as _pathlib
import typing as _typing
import numpy as _np

Array = _typing.TypeVar('Array', _np.ndarray)
ArrayOrStr = _typing.TypeVar('ArrayOrStr', str, _np.ndarray)
IterOrNone = _typing.TypeVar('IterOrNone', _typing.Iterable, None)
FloatOrNone = _typing.TypeVar('FloatOrNone', float, None)

ParserType = _typing.Tuple[_typing.Dict[str, _typing.Any], _typing.List[str]]
PathType = _typing.TypeVar('PathType', str, _pathlib.Path)
PathGen = _typing.Generator[PathType, None, None]
