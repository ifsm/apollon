# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

# apollon/som/som.py
# SelfOrganizingMap module
#

import itertools
import numpy as _np
import matplotlib.pyplot as _plt
from scipy import stats as _stats
from scipy.spatial import distance as _distance

from apollon.io.io import save as _save
from apollon.som import defaults as _defaults
from . import neighbors as _neighbors
from . import utilities as _som_utils
from .. types import Array as _Array
from apollon.aplot import _new_axis, _new_axis_3d
from apollon import aplot as aplot


class _SomBase:
    def __init__(self, dims, n_iter, eta, nhr, nh_shape, init_distr, metric, mode, seed=None):

        # check dimensions
        for d in dims:
            if not isinstance(d, int) or not d >= 1:
                raise ValueError('Dimensions must be integer > 0.')

        self.dims = dims
        self.dx, self.dy, self.dw = self.dims
        self.shape = (self.dx, self.dy)
        self.n_units = self.dx * self.dy
        self.center = self.dx // 2, self.dy // 2
        self.whist = _np.zeros(self.n_units)
        self.n_iter = n_iter
        self.mode = mode
        self.metric = metric
        self.isCalibrated = False
        self.calibration = None
        self.quantization_error = _np.zeros(n_iter)

        try:
            self._neighbourhood = getattr(_neighbors, nh_shape)
        except AttributeError:
            raise AttributeError(f'Neiborhood shape {nh_shape} is unknown. Use'
                        'one `gaussian`, `mexican`, `rect`, or `star`')

        # check training parameters
        if eta is None:
            self.init_eta = None
        else:
            if (0 <= eta <= 1.):
                self.init_eta = eta
            else:
                raise ValueError('eta not in [0, 1]')

        if nhr > 1:
            self.init_nhr = nhr
            #self.final_nhr = max(self.dx, self.dy) / _defaults.nhr_scale_factor
        else:
            raise ValueError('Neighbourhood radius must be int > 0.')

        if seed is not None:
            _np.random.seed(seed)

        if init_distr == 'uniform':
            self.weights = _np.random.uniform(0, 1, size=(self.n_units, self.dw))
        elif init_distr == 'simplex':
            self.weights = _som_utils.init_simplex(self.dw, self.n_units)
        elif init_distr == 'pca':
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown initializer "{init_distr}". Use'
                             '"uniform", "simplex", or "pca".')

        self._grid = _som_utils.grid(self.dx, self.dy)


    def calibrate(self, data, targets):
        """Retriev for every map unit the best matching vector of the input
        data set. Save its target value at the map units position on a
        new array called `calibration`.

        Args:
            data:     Input data set.
            targets:  Target labels.
        """
        bmiv, err = self.get_winners(data, argax=0)
        self._cmap = targets[bmiv]
        self.isCalibrated = True



    def save(self, path):
        """Save som object to file using pickle.

        Args:
            path: Save SOM to this path.
        """
        _save(self, path)


    def transform(self, data, flat=True):
        """Transform input data to feature space.

        Args:
            data:  2d array of shape (N_vect, N_features).
            flat:  Return flat index of True else 2d multi index.

        Returns:
            Position of each data item in the feature space.
        """
        bmu, err = self.get_winners(data)

        if flat:
            return bmu

        else:
            midx = _np.unravel_index(bmu, (self.dx, self.dy))
            return _np.array(midx)


class BatchMap(_SomBase):
    def __init__(self, dims: tuple, n_iter: int, eta: float, nhr: float,
                 nh_shape: str = 'gaussian', init_distr: str = 'uniform',
                 metric: str = 'euclidean', seed: int = None):

        super().__init__(dims, n_iter, eta, nhr, nh_shape, init_distr, metric,
                         mode='batch', seed=seed)


class IncrementalMap(_SomBase):
    def __init__(self, dims: tuple, n_iter: int, eta: float, nhr: float,
                 nh_shape: str = 'gaussian', init_distr: str = 'uniform',
                 metric: str = 'euclidean', seed: int = None):

        super().__init__(dims, n_iter, eta, nhr, nh_shape, init_distr, metric,
                         mode='incremental', seed=seed)


    def fit(self, train_data, verbose=False):
        eta_ = _som_utils.decrease_linear(self.init_eta, self.n_iter, _defaults.final_eta)
        nhr_ = _som_utils.decrease_expo(self.init_nhr, self.n_iter, _defaults.final_nhr)
        iter_ = range(self.n_iter)

        for (c_iter, c_eta, c_nhr) in zip(iter_, eta_, nhr_):
            if verbose:
                print('iter: {:2} -- eta: {:<5} -- nh: {:<6}' \
                 .format(c_iter, _np.round(c_eta, 4), _np.round(c_nhr, 5)))

            for fvect in _np.random.permutation(train_data):
                bmu, err = _som_utils.best_match(self.weights, fvect, self.metric)
                self.whist[bmu] += 1
                self.quantization_error[c_iter] += err

                m_idx= _np.atleast_2d(_np.unravel_index(bmu, self.shape)).T
                neighbors = self._neighbourhood(self._grid, m_idx, c_nhr)
                self.weights += c_eta * neighbors * (fvect - self.weights)


class SelfOrganizingMap(_SomBase):

    def __init__(self, dims: tuple, n_iter: int, eta: float, nhr: float,
                 nh_shape: str = 'gaussian', init_distr: str = 'uniform',
                 metric: str = 'euclidean', mode: str = 'incremental',
                 seed: int = None):

        super().__init__(dims, n_iter, eta, nhr, nh_shape, init_distr, metric, mode, seed)


    def _batch_update(self, data_set, c_nhr):
        # get bmus for vector in data_set
        bm_units, total_qE = self.get_winners(data_set)
        self.quantization_error.append(total_qE)

        # get bmu's multi index
        bmu_midx = _np.unravel_index(bm_units, self.shape)

        w_nh = _np.zeros((self.n_units, 1))
        w_lat = _np.zeros((self.n_units, self.dw))

        for bx, by, fv in zip(*bmu_midx, data_set):
            # TODO:  Find a way for faster nh computation
            c_nh = self._neighbourhood((bx, by), c_nhr)
            w_nh += c_nh
            w_lat += c_nh * fv

        self.weights = w_lat / w_nh


    def train_batch(self, data, verbose=False):
        """Feed the whole data set to the network and update once
           after each iteration.

        Args:
            data:    Input data set.
            verbose: Print verbose messages if True.
        """
        # main loop
        for (c_iter, c_nhr) in \
            zip(range(self.n_iter),
                _som_utils.decrease_linear(self.init_nhr, self.n_iter)):

            if verbose:
                print(c_iter, end=' ')

            self._batch_update(data, c_nhr)


    def predict(self, data):
        """Predict a class label for each item in input data. SOM needs to be
        calibrated in order to predict class labels.
        """
        if self.isCalibrated:
            midx = self.transform(data)
            return self._cmap[midx]
        else:
            raise AttributeError('SOM is not calibrated.')



class DotSom(_SomBase):
    def __init__(self, dims=(10, 10, 3), eta=.8, nh=8, n_iter=10,
                 metric='euclidean', mode=None, init_distr='uniform', seed=None):
        """ This SOM assumes a stationary PoissonHMM on each unit. The weight vector
        represents the HMMs distribution parameters in the following order
        [lambda1, ..., lambda_m, gamma_11, ... gamma_mm]

        Args:
            dims    (tuple) dx, dy, m
        """
        super().__init__(dims, eta, nh, n_iter, metric, mode, init_distr, seed)
        self._neighbourhood = self.nh_gaussian_L2

    def get_winners(self, data, argax=1):
        """Get the best matching neurons for every vector in data.

        Args:
            data:  Input data set
            argax: Axis used for minimization 1=x, 0=y.

        Returns:
            Indices of bmus and min dists.
        """
        # TODO: if the distance between an input vector and more than one lattice
        #       neuro is the same, choose winner randomly.

        d = _np.inner(data, self.weights)
        return _np.argmax(d), 0



    def fit(self, data, verbose=True):
        for (c_iter, c_eta, c_nhr) in \
            zip(range(self.n_iter),
                _som_utils.decrease_linear(self.init_eta, self.n_iter, _defaults.final_eta),
                _som_utils.decrease_expo(self.init_nhr, self.n_iter, _defaults.final_nhr)):

            if verbose:
                print('iter: {:2} -- eta: {:<5} -- nh: {:<6}' \
                 .format(c_iter, _np.round(c_eta, 4), _np.round(c_nhr, 5)))

            # always shuffle data
            self._incremental_update(_np.random.permutation(data), c_eta, c_nhr)


    def _incremental_update(self, data_set, c_eta, c_nhr):
        total_qE = 0
        for fv in data_set:
            bm_units, c_qE = self.get_winners(fv)
            total_qE += c_qE

            # update activation map
            self.whist[bm_units] += 1

            # get bmu's multi index
            bmu_midx = _np.unravel_index(bm_units, self.shape)

            # calculate neighbourhood over bmu given current radius
            c_nh = self._neighbourhood(bmu_midx, c_nhr)

            # update lattice
            u = self.weights + c_eta * fv
            self.weights = u / _distance.norm(u)

        self.quantization_error.append(total_qE)
