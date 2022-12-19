""" apollon/container.py -- Container Classes.

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net

Classes:
    FeatureSpace
    Params
"""
import csv
from dataclasses import dataclass, asdict
import json
import pathlib
import sys
from typing import (Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar)

import jsonschema

from . import io
from . types import Schema, PathType


GenericParams = TypeVar('GenericParams', bound='Parent')

@dataclass
class Params:
    """Parmeter base class."""
    _schema: ClassVar[Schema] = {}

    @property
    def schema(self) -> dict:
        """Returns the serialization schema."""
        return self._schema

    @classmethod
    def from_dict(cls: Type[GenericParams], instance: dict) -> GenericParams:
        """Construct Params from dictionary"""
        return cls(**instance)

    def to_dict(self) -> dict:
        """Returns parameters as dictionary."""
        return asdict(self)

    def to_json(self, path: PathType) -> None:
        """Write parameters to JSON file.

        Args:
            path:  File path.
        """
        instance = self.to_dict()
        jsonschema.validate(instance, self.schema, jsonschema.Draft7Validator)
        with pathlib.Path(path).open('w') as fobj:
            json.dump(instance, fobj)
