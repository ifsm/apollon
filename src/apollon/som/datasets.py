"""apollon/som/datasets.py

Licensed under the terms of the BSD-3-Clause license.
Copyright (C) 2019 Michael Blaß
mblass@posteo.net

Function for generating test and illustration data sets.
"""
from typing import Optional, Tuple

import numpy as np
from scipy import stats

def norm_circle(n_classes: int, n_per_class: int, class_std: int,
                center: Tuple[int, int] = (0, 0), radius: int = 5,
                seed: Optional[int] = None):
    """Generate ``n_per_class`` samples from ``n_classes`` bivariate normal
    distributions, each with standard deviation ``class_std``. The means
    are equidistantly placed on a circle with radius ``radius``.

    Args:
        n_classes:    Number of classes.
        n_per_class:  Number of samples in each class.
        class_std:    Standard deviation for every class.
        center:       Center of ther circle.
        radius:       Radius of the circle on which the means are placed.
        seed:         Set the random seed.

    Returns:
        Data set and target vector.
    """
    n_samples = n_classes * n_per_class
    ang = np.pi * np.linspace(0, 360, n_classes, endpoint=False) / 180
    xy_pos = np.stack((np.sin(ang), np.cos(ang)), axis=1)
    xy_pos *= radius + np.asarray(center)

    out = np.empty((n_samples, 2))
    for i, pos in enumerate(xy_pos):
        idx = slice(i*n_per_class, (i+1)*n_per_class)
        distr = stats.multivariate_normal(pos, np.sqrt(class_std), seed=seed)
        out[idx, :] = distr.rvs(n_per_class)
    return out, np.arange(n_samples) // n_per_class
