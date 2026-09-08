"""
General JSON IO
"""

import json
import pathlib
from typing import Any

import numpy as np

from .. typing import Array, PathType


def dump_json(obj: Any, path: PathType) -> None:
    """Write ``obj`` to JSON file.

    This function can handle numpy arrays.

    Args:
        obj:   Object to be encoded.
        path:  Output file path.
    """
    path = pathlib.Path(path)
    with path.open('w', encoding='utf-8') as json_file:
        json.dump(obj, json_file, cls=ArrayEncoder)


def load_json(path: PathType) -> Any:
    """Load JSON file.

    Args:
        path: Path to file.

    Returns:
        Decoded JSON content, with any encoded numpy arrays restored.
    """
    path = pathlib.Path(path)
    with path.open('r', encoding='utf-8') as fobj:
        return json.load(fobj, object_hook=_ndarray_hook)


def validate_ndarray(obj: dict[str, Any]) -> bool:
    """Check whether ``encoded_arr`` is a valid instance of
    ``ndarray.schema.json``.

    Args:
        obj:  Instance to validate.

    Returns:
        ``True``, if instance is valid.
    """
    return (
        "__ndarray__" in obj
        and obj['__ndarray__']
        and "__dtype__" in obj
        and "data" in obj
    )


def decode_ndarray(instance: dict[str, Any]) -> Array:
    """Decode numerical numpy arrays from a JSON data stream.

    Args:
        instance:  Instance of ``ndarray.schema.json``.

    Returns:
        Numpy array.

    Raises:
        TypeError: If ``instance`` is not a valid instance of
            ``ndarray.schema.json``.
    """
    if validate_ndarray(instance):
        dtype = instance['__dtype__']
        if np.dtype(dtype).kind == 'c':
            out = np.empty(np.shape(instance['data']['real']), dtype=dtype)
            out.real = instance['data']['real']
            out.imag = instance['data']['imag']
            return out
        return np.array(instance['data'], dtype=dtype)
    raise TypeError(
        f"{instance!r} is not a valid ndarray instance: missing or "
        "invalid '__ndarray__', '__dtype__', or 'data' field"
    )


def encode_ndarray(arr: Array) -> dict[str, Any]:
    """Transform an numpy array to a JSON-serializable dict.

    Array must have a numerical dtype, including complex. Datetime
    objects are currently not supported.

    Args:
        arr:  Numpy ndarray.

    Returns:
        JSON-serializable dict adhering ``ndarray.schema.json``.
    """
    if np.iscomplexobj(arr):
        data: Any = {'real': arr.real.tolist(), 'imag': arr.imag.tolist()}
    else:
        data = arr.tolist()
    return {'__ndarray__': True, '__dtype__': arr.dtype.str, 'data': data}


def _ndarray_hook(inp: dict[str, Any]) -> Array | dict[str, Any]:
    try:
        return decode_ndarray(inp)
    except TypeError:
        return inp


class ArrayEncoder(json.JSONEncoder):
    """Encode np.ndarrays to JSON.

    Simply set the ``cls`` parameter of the dump method to this class.
    """
    def default(self, o: Any) -> Any:
        """Custom JSON encoder for numpy arrays. Other types are passed
        on to ``JSONEncoder.default``.

        Args:
            o:  Object to encode.

        Returns:
            JSON-serializable dictionary.
        """
        if isinstance(o, np.ndarray):
            return encode_ndarray(o)
        return json.JSONEncoder.default(self, o)
