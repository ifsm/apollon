#!/usr/bin/env python3

"""
hmm_base.py -- Base class of the apollon Hidden Markov Model.
Copyright (C) 2017  Michael Blaß

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


import numpy as _np
import apollon


class HMM_Base:
    """The Apollon Hidden Markov Model base class implements behaviour
       common for all types of HMMs.

       Coding convention:
           `_paramname` means an initial guess for a parameter.
           `paramname_` represents the estimated parameter.
    """

    def __init__(self, m, _gamma=None, _delta=None, verbose=True):

        if isinstance(m, int):
            self.m = m
        else:
            raise ValueError('Number of states must be integer.')

        self.verbose            = verbose
        self.trained            = False
        self.training_date      = ''
        self.apollon_version    = apollon.__version__

        self.nll = 0
        self.aic = 0
        self.bic = 0

    def compute_aic(self) -> float:
        """Compute the Akaike Information Criterion (AIC) of the current model.

        Returns:
            (float) AIC
        """
        return 2*self.nll + 2*(self.m * self.m + self.m)

    def compute_bic(self, n: int) -> float:
        """Compute the Bayesian Information Criterion (BIC) of the current model.

        Params:
            n   (int)   Number of training observations.
        Returns:
            (flaot) BIC
        """
        return 2*self.nll + (self.m * self.m + self.m) * _np.log(n)
