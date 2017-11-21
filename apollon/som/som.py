#!python3
# -*- coding: utf-8 -*-

# apollon/som/som.py
# SelfOrganizingMap module
#

import numpy as _np
from scipy import stats as _stats
from scipy.spatial import distance as _distance

from apollon.IO import save as _save
from apollon.som import utilities as _utilities
from apollon.som import defaults as _defaults
from apollon.decorators import switch_interactive
from apollon.aplot import _new_figure, _new_axis, _new_axis_3d


class _som_base:

    def __init__(self, dims=(10, 10, 3), eta=.8, nhr=5,
                 metric='euclidean', init_distr='simplex'):

        # check dimensions
        for d in dims:
            if not isinstance(d, int) or not d >= 1:
                raise ValueError('Dimensions must be integer > 0.')

        self.shape = dims
        self.dx = self.shape[0]
        self.dy = self.shape[1]
        self.dw = self.shape[2]
        self.n_N = self.dx * self.dy    # number of neurons

        self.center = self.dx // 2, self.dy // 2

        # check parameters
        if eta is None:
            self.init_eta = None
        else:
            if (0 <= eta <= 1.):
                self.init_eta = eta
            else:
                raise ValueError('eta not in [0, 1]')

        if isinstance(nhr, int) and nhr > 1:
            self.init_nhr = nhr
            self.final_nhr = max(self.dx, self.dy) / _defaults.nhr_scale_factor
        else:
            raise ValueError('Neighbourhood radius must be int > 0.')

        self.metric = metric

        # Init weights
        _np.random.seed(1)

        if init_distr == 'uniform':
            self.weights = _np.random.uniform(0, 1, size=(self.n_N, self.dw))
        elif init_distr == 'simplex':
            self.weights = self._init_st_mat()

        # Allocate array for winner histogram
        # TODO: add array to collect for every winner the correspondig inp vector.
        self.whist = _np.zeros(self.n_N)

        # grid data for neighbourhood calculation
        self._grid = _np.dstack(_np.mgrid[:self.dx, :self.dy])

        # calibration
        self.isCalibrated = False
        self.calibration = None

        # measures
        self.quantization_error = []


    def get_winners(self, data, argax=1):
        '''Get the best matching neurons for every vector in data.

           Params:
                data    (np.array) Input data set
                argax   (int) Axis used for minimization 1=x, 0=y.

            Return:
                (np.array, np.ndarray) Indices of bmus and min dists.
        '''
        # TODO: if the distance between an input vector and more than one lattice
        #       neuro is the same, choose winner randomly.

        if data.ndim == 1:
            d = _distance.cdist(data[None, :], self.weights, metric=self.metric)
            return _np.argmin(d), _np.min(d)**2
        elif data.ndim == 2:
            ds = _distance.cdist(data, self.weights, metric=self.metric)
            return _np.argmin(ds, axis=argax), _np.sum(_np.min(ds, axis=argax)**2)
        else:
            raise ValueError('Wrong dimension of input data: {}'.format(data.ndim))


    def _neighbourhood(self, point, nhr):
        var = _stats.multivariate_normal(mean=point, cov=((nhr, 0), (0, nhr)))
        out = var.pdf(self._grid)
        return (out / _np.max(out)).reshape(self.n_N, 1)


    def _init_st_mat(self):
        '''Initialize the weights with stochastic matrices.

        The rows of each n by n stochastic matrix are sampes drawn from the
        Dirichlet distribution, where n is the number of rows and cols of the
        matrix. The diagonal elemets of the matrices are set to twice the
        probability of the remainigne elements.
        The square root n of the weight vectors' size must be element of the
        natural numbers, so that the weight vector is reshapeable to a sqare
        matrix.
        '''
        # check for square matrix
        d = _np.sqrt(self.dw)
        is_not_qm = bool(d - int(d))
        if is_not_qm:
            raise ValueError('Weight vector (len={}) must be reshapeable to square matrix.'.format(self.dw))
        else:
            d = int(d)

        # set alpha
        alpha = _np.full((d, d), 500)
        _np.fill_diagonal(alpha, 1000)

        # sample from dirichlet distributions
        st_matrix = _np.hstack(_stats.dirichlet.rvs(alpha=a, size=self.n_N)
                               for a in alpha)
        return st_matrix


    def calibrate(self, data, targets):
        '''Retriev for every map unit the best matching vector of the input
           data set. Save its target value at the map units position on a
           new array called `calibration`.

           Params:
            data    (2d array) Input data set.
            targets (1d array) Target labels.
        '''
        bmiv, err = self.get_winners(data, argax=0)
        self._cmap = targets[bmiv]
        self.isCalibrated = True


    @switch_interactive
    def plot_calibration(self, lables=None, ax=None, **kwargs):
        # TODO: add params to docstring
        '''Plot calibrated map.'''
        if not self.isCalibrated:
            raise ValueError('Map not calibrated.')
        else:
            if ax is None:
                ax = _new_axis(xlim=(0, self.dx), ylim=(0, self.dy), **kwargs)
            ax.imshow(self._cmap.reshape(self.dx, self.dy))
            #return ax


    @switch_interactive
    def plot_datamap(self, data, targets, interp='None', marker=False, **kwargs):
        '''Represent the input data on the map by retrieving the best
           matching unit for every element in `data`. Mark each map unit
           with the corresponding target value.

           Params:
                data    (2d-array) Input data set.
                targets (array) Class labels or values.
                interp  (str) matplotlib interpolation method name.
                marker  (bool) Plot markers in bmu position if True.

           Return:
                (AxesSubplot) axis, umatrix, bmu_xy
        '''
        ax, udm = self.plot_umatrix(interp=interp, **kwargs)
        bmu, err = self.get_winners(data)

        x, y = _np.unravel_index(bmu, (self.dx, self.dy))
        fd = {'color':'#cccccc'}
        if marker:
            ax.scatter(y, x, s=40, marker='x', color='r')

        for i, j, t in zip(x, y, targets):
            ax.text(j, i, t, fontdict=fd,
                    horizontalalignment='center',
                    verticalalignment='center')
        return (ax, udm, (x, y))


    @switch_interactive
    def plot_umatrix(self, w=1, interp='None', ax=None, **kwargs):
        '''Plot the umatrix. The color on each unit (x, y) represents its
           mean distance to all direct neighbours.

           Params:
               w        (int) Neighbourhood width.
               interp   (str) matplotlib interpolation method name.
               ax       (plt.Axis) Provide custom axis object.

           Return:
               (AxesSubplot, np.array) the axis, umatrix
        '''
        if ax is None:
            ax = _new_axis(xlim=(0, self.dx), ylim=(0, self.dy), **kwargs)
        udm = _utilities.umatrix(self.weights, self.dx, self.dy, w=w)
        ax.imshow(udm, interpolation=interp)
        return ax, udm


    @switch_interactive
    def plot_umatrix3d(self, w=1, **kwargs):
        '''Plot the umatrix in 3d. The color on each unit (x, y) represents its
           mean distance to all direct neighbours.

           Params:
               w            (int) Neighbourhood width.
               **kwargs     Pass keywors to _new_axis_3d.
           Return:
               (Axes3DSubplot, np.array) the axis, umatrix
        '''
        fig, ax = _new_axis_3d(**kwargs)
        udm = _utilities.umatrix(self.weights, self.dx, self.dy, w=w)
        X, Y = _np.mgrid[:self.dx, :self.dy]
        ax.plot_surface(X, Y, udm, cmap='viridis')
        return ax, udm


    @switch_interactive
    def plot_variables(self, interp='None', titles=True, axison=False, **kwargs):
        '''Represent the influence of each variable of the input data on the
        som lattice as heat map.

        Params:
            interp     (str) matplotlib interpolation method name.
            titles     (bool) Print variable above each heatmap.
            axison     (bool) Plot spines if True.
        '''
        _z = _np.sqrt(self.dw)
        _iz = int(_z)

        if _z % 2 == 0:
            if _z == 1:
                x = _iz
                y = self.dw
            else:
                x = y = _iz
        else:
            x = _iz
            y = self.dw - x**2

        fig = _new_figure(**kwargs)
        for i in range(1, self.dw+1):
            ax = _new_axis(fig=fig, sp_pos=(x, y, i), axison=axison)
            ax.imshow(self.weights[:,i-1].reshape(self.dx, self.dy),
                      interpolation=interp)
            if titles:
                ax.set_title(str(i))


    @switch_interactive
    def plot_whist(self, interp='None', ax=None, **kwargs):
        '''Plot the winner histogram.

           The darker the color on position (x, y) the more often neuron (x, y)
           was choosen as winner. The number of winners at edge neuros is
           magnitudes of order higher than on the rest of the map. Thus, the
           histogram is shown in log-mode.

           Params:
               interp    (str) matplotlib interpolation method name.
               ax        (plt.Axis) Provide custom axis object.

           Return:
               (AxesSubplot) the axis.
        '''
        if ax is None:
            ax = _new_axis(xlim=(0, self.dx), ylim=(0, self.dy), **kwargs)
        ax.imshow(_np.log1p(self.whist.reshape(self.dx, self.dy)),
                  vmin=0, cmap='Greys', interpolation=interp)
        return ax


    def save(self, path):
        '''Save som object to file using pickle.

        Params:
            path    (str) Save SOM to this path.
        '''
        _save(self, path)


class SelfOrganizingMap(_som_base):
    def __init__(self, dims=(10, 10, 3), eta=.8, nh=5,
                 metric='euclidean', init_distr='simplex'):
        super().__init__(dims, eta, nh, metric, init_distr)


    def _incremental_update(self, data_set, c_eta, c_nhr):
        total_qE = 0
        for fv in data_set:
            bm_units, c_qE = self.get_winners(fv)
            total_qE += c_qE

            # update activation map
            self.whist[bm_units] += 1

            # get bmu's multi index
            bmu_midx = _np.unravel_index(bm_units, (self.shape[0], self.shape[1]))

            # calculate neighbourhood over bmu given current radius
            c_nh = self._neighbourhood(bmu_midx, c_nhr)

            # update lattice
            self.weights += c_eta * c_nh * (fv - self.weights)
        self.quantization_error.append(total_qE)


    def _batch_update(self, data_set, c_nhr):
        # get bmus for vector in data_set
        bm_units, total_qE = self.get_winners(data_set)
        self.quantization_error.append(total_qE)

        # get bmu's multi index
        bmu_midx = _np.unravel_index(bm_units, (self.shape[0], self.shape[1]))

        w_nh = _np.zeros((self.n_N, 1))
        w_lat = _np.zeros((self.n_N, self.dw))

        for bx, by, fv in zip(*bmu_midx, data_set):
            # TODO:  Find a way for faster nh computation
            c_nh = self._neighbourhood((bx, by), c_nhr)
            w_nh += c_nh
            w_lat += c_nh * fv

        self.weights = w_lat / w_nh


    def train_batch(self, data, N_iter, verbose=False):
        '''Feed the whole data set to the network and update once
           after each iteration.


           Params:
               data    (2d array) Input data set.
               N_iter  (int) Number of iterations to perform.
               verbose (bool) Print verbose messages if True.'''
        # main loop
        for (c_iter, c_nhr) in \
            zip(range(N_iter),
                _utilities.decrease_linear(self.init_nhr, N_iter)):

            if verbose:
                print(c_iter, end=' ')

            self._batch_update(data, c_nhr)


    def train_minibatch(self, data, N_iter, verbose=False):
        raise NotImplementedError


    def train_incremental(self, data, N_iter, verbose=False):
        '''Randomly feed the data to the network and update after each
           data item.

           Params:
               data    (2d array) Input data set.
               N_iter  (int) Number of iterations to perform.
               verbose (bool) Print verbose messages if True.'''
        # main loop
        for (c_iter, c_eta, c_nhr) in \
            zip(range(N_iter),
                _utilities.decrease_linear(self.init_eta, N_iter, _defaults.final_eta),
                _utilities.decrease_linear(self.init_nhr, N_iter, self.final_nhr)):

            if verbose:
                print('iter: {:2} -- eta: {:<5} -- nh: {:<6}' \
                 .format(c_iter, _np.round(c_eta, 4), _np.round(c_nhr, 5)))

            # always shuffle data
            self._incremental_update(_np.random.permutation(data), c_eta, c_nhr)


from apollon.hmm.poisson_hmm import hmm_distance
class PoissonHmmSom(_som_base):
    def __init__(self, dims=(10, 10, 2), eta=.8, nh=5,
                 metric=hmm_distance, init_distr='simplex'):
        """
        This SOM assumes a stationary PoissonHMM on each unit. The weight vector
        represents the HMMs distribution parameters in the following order
        [lambda1, ..., lambda_m, gamma_11, ... gamma_mm]
        Params:
            dims    (tuple) dx, dy, m
        """
        dx, dy, m = dims
        dw = m * m + m
        dims = dx, dy, dw
        super().__init__(dims, eta, nh, metric, init_distr)


    def get_winners(self, data, argax=1):
        '''Get the best matching neurons for every vector in data.

           Params:
                data    (np.array) Input data set
                argax   (int) Axis used for minimization 1=x, 0=y.

            Return:
                (np.array, np.ndarray) Indices of bmus and min dists.
        '''
        # TODO: if the distance between an input vector and more than one lattice
        #       neuro is the same, choose winner randomly.

        if data.ndim == 1:
            d = _distance.cdist(data[None, :], self.weights, metric=self.metric)
            return _np.argmin(d), _np.min(d)**2
        elif data.ndim == 2:
            ds = _distance.cdist(data, self.weights, metric=self.metric)
            return _np.argmin(ds, axis=argax), _np.sum(_np.min(ds, axis=argax)**2)
        else:
            raise ValueError('Wrong dimension of input data: {}'.format(data.ndim))



    def _incremental_update(self, data_set, c_eta, c_nhr):
        total_qE = 0
        for fv in data_set:
            bm_units, c_qE = self.get_winners(fv)
            total_qE += c_qE

            # update activation map
            self.whist[bm_units] += 1

            # get bmu's multi index
            bmu_midx = _np.unravel_index(bm_units, (self.shape[0], self.shape[1]))

            # calculate neighbourhood over bmu given current radius
            c_nh = self._neighbourhood(bmu_midx, c_nhr)

            # update lattice
            self.weights += c_eta * c_nh * (fv - self.weights)
        self.quantization_error.append(total_qE)
