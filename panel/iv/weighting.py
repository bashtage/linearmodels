from __future__ import print_function, absolute_import, division

from numpy import nonzero, argsort, zeros, r_

from panel.iv.covariance import KERNEL_LOOKUP


class HomoskedasticWeightMatrix(object):
    """
    Parameters
    ----------
    **weight_config
        Keywords to pass to weight matrix
    """

    def __init__(self, center=False):
        self._center = center
        self._bandwidth = 0

    def weight_matrix(self, x, z, eps):
        nobs = x.shape[0]
        mu = eps.mean(0) if self._center else 0
        s2 = (eps - mu).T @ (eps - mu) / nobs
        return s2 * z.T @ z / nobs

    @property
    def config(self):
        return {'center': self._center}


class HeteroskedasticWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, center=False):
        super(HeteroskedasticWeightMatrix, self).__init__(center)

    def weight_matrix(self, x, z, eps):
        nobs = x.shape[0]
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        return ze.T @ ze / nobs


class KernelWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, center=False, kernel='bartlett', bw=None):
        super(KernelWeightMatrix, self).__init__(center)
        self._bandwidth = bw
        self._kernel = kernel
        self._kernels = KERNEL_LOOKUP

    def weight_matrix(self, x, z, eps):
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu
        nobs, ninstr = z.shape

        # TODO: Fix this to allow optimal bw selection by default

        bw = self._bandwidth if self._bandwidth is not None else nobs - 2
        w = self._kernels[self._kernel](bw)
        s = ze.T @ ze
        for i in range(1, bw + 1):
            s += w[i] * ze[i:].T @ ze[:-i]
        return s / nobs

    def _optimal_bandwidth(self, x, z, eps):
        # TODO: Implement this
        pass

    @property
    def config(self):
        return {'center': self._center,
                'bw': self._bandwidth,
                'kernel': self._kernel}


class OneWayClusteredWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, center=False, clusters=None):
        super(OneWayClusteredWeightMatrix, self).__init__(center)
        self._clusters = clusters

    def weight_matrix(self, x, z, eps):
        wc = self.config
        nobs, ninstr = z.shape

        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        clusters = self._clusters
        if clusters is None:
            raise ValueError('clusters must be provided')
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
    def config(self):
        return {'center': self._center,
                'clusters': self._clusters}
