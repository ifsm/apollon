# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

"""
apollon/_defaults.py --  Defaut definitions
"""
import pathlib

from . import APOLLON_PATH


SCHEMA_EXT = '.schema.json'
SCHEMA_DIR_PATH = pathlib.Path(APOLLON_PATH).parent.joinpath('schema')

DATE_TIME = '%Y-%m-%d %H:%M:%S'

SPL_REF = 2e-5

PP_SIGNAL = {'linewidth': 1, 'linestyle': 'solid', 'color': 'k', 'alpha': .5,
             'zorder': 0}

PP_SIG_ONS = {'linewidth': 2, 'linestyle': 'solid', 'color': 'C1',
              'alpha': .9, 'zorder': 0}

PP_ONSETS = {'linewidth': 3, 'linestyle': 'dashed', 'color': 'C1', 'alpha': .9,
             'zorder': 10}
