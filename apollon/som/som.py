#!python3
# -*- coding: utf-8 -*-

# apollon/som/som.py
# SelfOrganizingMap module
#

import numpy as _np
from scipy import stats as _stats
from scipy.spatial import distance as _distance

from apollon.som import utilities as _utilities


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


    def get_winners(self, data, argax=1):
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


    def plot_whist(self):
        # TODO: integrate aplot._new_figure()
        plt.imshow(self.whist.reshape(self.dx, self.dy),
               vmin=0, cmap='Greys', interpolation='None')
        # TODO: use decorator of aplot instead for mode switching.
        self._switchInteractive()


    def plot_umatrix(self, w=1, ax=None):
        # TODO: integrate aplot._new_figure()
        udm = _utilities.umatrix(self.lattice, self.dx, self.dy, w=w)
        if ax is None:
            ax = _plt.gca()
        ax.imshow(udm)
        # TODO: use decorator of aplot instead for mode switching.
        self._switchInteractive()


    @staticmethod
    def _switchInteractive():
        if not plt.isinteractive():
            plt.show()

    def calibrate(self, data, targets):
        '''Retrieve the best matching unit for every element
           in `data` and mark it with the corresponding target value.
        '''
        bmu = self.get_winners(data)
        x, y = __np.unravel_index(bmu, (self.shape[0], self.shape[1]))
        fig, ax = plt.subplots(1)
        self.plot_umatrix(ax=ax)
        # TODO: align text to center of rects of umatrix plot
        for i,j,t in zip(x,y, targets):
            ax.text(j,i,t)
        # TODO: use decorator of aplot instead for mode switching.
        self._switchInteractive()

    # TODO: does not work
    def cluster(self, data, targets):
        '''Retriev the best matching element of data for every
           map unit and save the corresponding target value in a
           new array.'''

        out = __np.zeros(self.n_N)
        bmiv = self.get_winners(data, argax=0)
        out[bmiv] = targets[bmiv]
        imshow(out.reshape(self.dx, self.dy))
        #return out


class SelfOrganizingMap(_som_base):
    def __init__(self, dims=(10, 10, 3), eta=.8, nh=5,
                 metric='euclidean', init_distr='simplex'):
        super().__init__(dims, eta, nh, metric, init_distr)



    def train_basic(self, data, N_iter, feed_rnd=True):

        if feed_rnd:
            data_set = _np.random.permutation(data)
        else:
            data_set = data

        for (c_iter, c_eta, c_nhr) in \
            zip(range(N_iter),
                _utilities.decrease_linear(self.init_eta, N_iter),
                _utilities.decrease_linear(self.init_nhr, N_iter)):

            # get bmus
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


    def map_response(self, data_i):
        return _distance.cdist(data_i[None, :], self.lattice)
