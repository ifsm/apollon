#!python3
# -*- coding: utf-8 -*-

# apollon/som/som.py
# SelfOrganizingMap module
#

import numpy as _np
from scipy import stats as _stats
from scipy.spatial import distance as _distance

from apollon.som import utilities as _utilities
from apollon.decorators import switch_interactive
from apollon.aplot import _new_figure, _new_axis


class _som_base:

    def __init__(self, dims=(10, 10, 3), eta=.8, nh=5,
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
        if 0 <= eta <= 1.:
            self.init_eta = eta
        else:
            raise ValueError('eta not in [0, 1]')

        if isinstance(nh, int) and nh > 1:
            self.init_nhr = nh
        else:
            raise ValueError('Neighbourhood radius must be int > 0.')

        if metric in _distance.__all__:
            self.metric = metric
        else:
            raise ValueError('Unknown metric.')

        # Init weights
        _np.random.seed(1)

        if init_distr == 'uniform':
            self.lattice = _np.random.uniform(0, 1, size=(self.n_N, self.dw))
        elif init_distr == 'simplex':
            self.lattice = _stats.dirichlet.rvs([1] * self.dw, self.n_N)

        # Allocate array for winner histogram
        # TODO: add array to collect for every winner the correspondig inp vector.
        self.whist = _np.zeros(self.n_N)

        # grid data for neighbourhood calculation
        self._grid = _np.dstack(_np.mgrid[0:dims[0], 0:dims[1]])

        # calibration
        self.isCalibrated = False
        self._cmap = None


    def get_winners(self, data, argax=1):
        '''Get the best matching neurons for every vector in data.

           Params:
                data    (np.array) Input data set
                argax   (int) Axis used for minimization 1=x, 0=y.

            Return:
                (np.array) Indices of bmus.
        '''
        # TODO: if the distance between an input vector and more than one lattice
        #       neuro is the same, choose winner randomly.

        if data.ndim == 1:
            d = _distance.cdist(data[None, :], self.lattice, metric=self.metric)
            return _np.argmin(d)
        elif data.ndim == 2:
            ds = _distance.cdist(data, self.lattice, metric=self.metric)
            return _np.argmin(ds, axis=argax)
        else:
            raise ValueError('Wrong dimension of input data: {}'.format(data.ndim))


    def _neighbourhood(self, point, nhr):
        var = _stats.multivariate_normal(mean=point, cov=((nhr, 0), (0, nhr)))
        out = var.pdf(self._grid)
        return (out / _np.max(out)).reshape(self.n_N, 1)


    def calibrate(self, data, targets):
        '''Retriev for every map unit the best matching vector of the input
           data set. Save its target value at the map units position on a
           new array called cmap (calibrate map).
        '''
        bmiv = self.get_winners(data, argax=0)
        self._cmap = targets[bmiv]
        self.isCalibrated = True


    @switch_interactive
    def plot_calibration(self, ax=None, **kwargs):
        # TODO: add params to docstring
        '''Plot calibrated map.'''
        if not self.isCalibrated:
            raise ValueError('Map not calibrated.')
        else:
            if ax is None:
                ax = _new_axis(xlim=(0, self.dx), ylim=(0, self.dy), **kwargs)
            ax.imshow(self._cmap.reshape(self.dx, self.dy))
            return ax


    @switch_interactive
    def plot_datamap(self, data, targets, interp='None', marker=True, **kwargs):
        '''Represent the input data on the map by retrieving the best
           matching unit for every elementin `data`. Mark each map unit
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
        bmu = self.get_winners(data)
        x, y = _np.unravel_index(bmu, (self.shape[0], self.shape[1]))

        fd = {'color':'#cccccc'}
        if marker:
            ax.scatter(y, x, s=40, marker='x', color='r')
        for i, j, t in zip(x, y, targets):
            ax.text(j, i, t, fontdict=fd,
                    horizontalalignment='center',
                    verticalalignment='center')

        return (ax, udm, (x,y))


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
            ax.imshow(self.lattice[:,i-1].reshape(self.dx, self.dy),
                      interpolation=interp)
            if titles:
                ax.set_title(str(i))


    @switch_interactive
    def plot_whist(self, interp='None', ax=None, **kwargs):
        '''Plot the winner histogram of the lattice. The darker color on
           position (x, y) the more often neuron (x, y) was choosen as winner.

           Params:
               interp    (str) matplotlib interpolation method name.
               ax        (plt.Axis) Provide custom axis object.

           Return:
               (AxesSubplot) the axis.
        '''
        if ax is None:
            ax = _new_axis(xlim=(0, self.dx), ylim=(0, self.dy), **kwargs)
        ax.imshow(self.whist.reshape(self.dx, self.dy),
               vmin=0, cmap='Greys', interpolation=interp)
        return ax


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
        udm = _utilities.umatrix(self.lattice, self.dx, self.dy, w=w)
        ax.imshow(udm, interpolation=interp)
        return (ax, udm)



class SelfOrganizingMap(_som_base):
    def __init__(self, dims=(10, 10, 3), eta=.8, nh=5,
                 metric='euclidean', init_distr='simplex'):
        super().__init__(dims, eta, nh, metric, init_distr)


    def _basic_update(self, data_set, c_eta, c_nhr):
        for fv in data_set:
            bm_units = self.get_winners(fv)

            # update activation map
            self.whist[bm_units] += 1

            # get bmu's multi index
            bmu_midx = _np.unravel_index(bm_units, (self.shape[0], self.shape[1]))

            # calculate neighbourhood over bmu given current radius
            c_nh = self._neighbourhood(bmu_midx, c_nhr)

            # update lattice
            self.lattice += c_eta * c_nh * (fv - self.lattice)


    def _batch_update(self, data_set, c_nhr):
        # get bmus for vector in data_set
        bm_units = self.get_winners(data_set)

        # get bmu's multi index
        bmu_midx = _np.unravel_index(bm_units, (self.shape[0], self.shape[1]))

        w_nh = _np.zeros((self.n_N, 1))
        w_lat = _np.zeros((self.n_N, self.dw))

        for bx, by, fv in zip(*bmu_midx, data_set):
            c_nh = self._neighbourhood((bx, by), c_nhr)
            w_nh += c_nh
            w_lat += c_nh * fv

        self.lattice = w_lat / w_nh


    def train_batch(self, data, N_iter):
        '''Train using batch algorithm.'''
        # main loop
        for (c_iter, c_nhr) in \
            zip(range(N_iter),
                _utilities.decrease_linear(self.init_nhr, N_iter)):

            # feed data in given order
            self._batch_update(data, c_nhr)


    def train_basic_order(self, data, N_iter):
        '''Train by feeding the data in the give order.'''
        # main loop
        for (c_iter, c_eta, c_nhr) in \
            zip(range(N_iter),
                _utilities.decrease_linear(self.init_eta, N_iter),
                _utilities.decrease_linear(self.init_nhr, N_iter)):

            # verbose
            #print(c_iter, end=' ')

            # feed data in given order
            self._basic_update(data, c_eta, c_nhr)


    def train_basic_rnd(self, data, N_iter):
        '''Train by shuffeling the dataset each iteration'''
        # main loop
        for (c_iter, c_eta, c_nhr) in \
            zip(range(N_iter),
                _utilities.decrease_linear(self.init_eta, N_iter),
                _utilities.decrease_linear(self.init_nhr, N_iter)):

            # verbose
            #print(c_iter, end=' ')

            # always shuffle data
            self._basic_update(_np.random.permutation(data), c_eta, c_nhr)
