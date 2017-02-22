from __future__ import print_function, absolute_import, division

from numpy import arange, nonzero, argsort, zeros, r_

from panel.iv.covariance import KERNEL_LOOKUP


class HomoskedasticWeightMatrix(object):
    """
    Parameters
    ----------
    **weight_config
        Keywords to pass to weight matrix
    """

    def __init__(self, **weight_config):
        for key in weight_config:
            if key not in self.defaults:
                raise ValueError('Unknown weighting matrix configuration '
                                 'parameter {0}'.format(key))
        wc = self.defaults
        wc.update(weight_config)
        self._config = wc
        self._bandwidth = 0

    def weight_matrix(self, x, z, eps):
        nobs = x.shape[0]
        s2 = eps.T @ eps / nobs
        return s2 * z.T @ z / nobs

    @property
    def defaults(self):
        return {'center': False}

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def config(self):
        return self._config


class HeteroskedasticWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, **weight_config):
        super(HeteroskedasticWeightMatrix, self).__init__(**weight_config)

    def weight_matrix(self, x, z, eps):
        nobs = x.shape[0]
        wc = self.config
        ze = z * eps
        mu = ze.mean(axis=0) if wc['center'] else 0
        ze -= mu

        return ze.T @ ze / nobs


class KernelWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, **weight_config):
        super(KernelWeightMatrix, self).__init__(**weight_config)
        self._bandwidth = 0
        self._kernels = KERNEL_LOOKUP

    def weight_matrix(self, x, z, eps):
        wc = self.config

        ze = z * eps
        mu = ze.mean(axis=0) if wc['center'] else 0
        ze -= mu
        nobs, ninstr = z.shape

        # TODO: Fix this to allow optimal bw selection by default

        bw = wc['bw'] if wc['bw'] is not None else nobs - 2
        kernel = wc['kernel']
        w = self._kernels[kernel](bw)
        s = ze.T @ ze
        for i in range(1, bw + 1):
            s += w[i] * ze[i:].T @ ze[:-i]
        return s / nobs

    def _optimal_bandwidth(self, x, z, eps):
        # TODO: Implement this
        pass

    @property
    def defaults(self):
        return {'kernel': 'bartlett',
                'center': False,
                'bw': None}


class OneWayClusteredWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, **weight_config):
        super(OneWayClusteredWeightMatrix, self).__init__(**weight_config)

    def weight_matrix(self, x, z, eps):
        wc = self.config
        nobs, ninstr = z.shape

        ze = z * eps
        mu = ze.mean(axis=0) if wc['center'] else 0
        ze -= mu

        clusters = wc['clusters']
        clusters = arange(nobs) if clusters is None else clusters
        ind = argsort(clusters)
        ze = ze[ind]
        clusters = clusters[ind]

        locs = nonzero(r_[True, clusters[1:] != clusters[:-1], True])[0]
        st, en = locs[:-1], locs[1:]

        s = zeros((ninstr, ninstr))
        for sloc, eloc in zip(st, en):
            zec = ze[sloc:eloc]
            s += zec.T @ zec

        return s / nobs

    @property
    def defaults(self):
        return {'clusters': None,
                'center': False}
