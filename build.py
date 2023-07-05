from typing import Any
from setuptools.extension import Extension
import numpy as np


ext_features = Extension('apollon.signal._features',
    sources = ['src/apollon/signal/cdim.c',
               'src/apollon/signal/correlogram.c',
               'src/apollon/signal/_features_module.c'],
    include_dirs = ['include', np.get_include()])

ext_som_dist = Extension('apollon.som._distance',
        sources = ['src/apollon/som/distance.c',
                   'src/apollon/som/_distance_module.c'],
        include_dirs = ['include', np.get_include()])


def build(setup_kwargs: dict[str, Any]) -> None:
    setup_kwargs.update({"ext_modules": [ext_features, ext_som_dist] })
