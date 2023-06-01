"""
General JSON IO
"""

import json
import pathlib
from typing import Any, Union, Mapping

import numpy as np

from .. _defaults import SCHEMA_DIR_PATH
from .. types import Array, PathType


def dump(obj: Any, path: PathType) -> None:
    """Write ``obj`` to JSON file.

    This function can handel numpy arrays.

    If ``path`` is None, this fucntion writes to stdout.  Otherwise, encoded
    object is written to ``path``.

    Args:
        obj:   Object to be encoded.
        path:  Output file path.
    """
    path = pathlib.Path(path)
    with path.open('w') as json_file:
        json.dump(obj, json_file, cls=ArrayEncoder)


def load(path: PathType):
    """Load JSON file.

    Args:
        path: Path to file.

    Returns:
        JSON file as FeatureSpace.
    """
    path = pathlib.Path(path)
    with path.open('r') as fobj:
        return json.load(fobj, object_hook=_ndarray_hook)


def validate_ndarray(obj: Mapping) -> bool:
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


def decode_ndarray(instance: Mapping) -> Array:
    """Decode numerical numpy arrays from a JSON data stream.

    Args:
        instance:  Instance of ``ndarray.schema.json``.

    Returns:
        Numpy array.
    """
    if validate_ndarray(instance):
        return np.array(instance['data'], dtype=instance['__dtype__'])
    raise TypeError("xx")


def encode_ndarray(arr: Array) -> dict:
    """Transform an numpy array to a JSON-serializable dict.

    Array must have a numerical dtype. Datetime objects are currently
    not supported.

    Args:
        arr:  Numpy ndarray.

    Returns:
        JSON-serializable dict adhering ``ndarray.schema.json``.
    """
    return {'__ndarray__': True, '__dtype__': arr.dtype.str,
            'data': arr.tolist()}


def _ndarray_hook(inp: dict) -> Union[Array, dict]:
    try:
        return decode_ndarray(inp)
    except TypeError:
        return inp


class ArrayEncoder(json.JSONEncoder):
    # pylint: disable=E0202
    # Issue: False positive for E0202 (method-hidden) #414
    # https://github.com/PyCQA/pylint/issues/414
    """Encode np.ndarrays to JSON.

    Simply set the ``cls`` parameter of the dump method to this class.
    """
    def default(self, inp: Any) -> Any:
        """Custon SON encoder for numpy arrays. Other types are passed
        on to ``JSONEncoder.default``.

        Args:
            inp:  Object to encode.

        Returns:
            JSON-serializable dictionary.
        """
        if isinstance(inp, Array):
            return encode_ndarray(inp)
        return json.JSONEncoder.default(self, inp)
