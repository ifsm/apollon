# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
Apollon feature extraction framework.
"""

import os as _os
import pkg_resources as _pkg


__version__ = _pkg.get_distribution('apollon').version

APOLLON_PATH = _os.path.dirname(_os.path.realpath(__file__))
