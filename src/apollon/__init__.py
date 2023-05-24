# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
Apollon feature extraction framework.
"""

import os as _os
from importlib import metadata as _meta

__version__ = _meta.version("apollon")

APOLLON_PATH = _os.path.dirname(_os.path.realpath(__file__))
