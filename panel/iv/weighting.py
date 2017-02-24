from __future__ import print_function, absolute_import, division

from numpy import nonzero, argsort, zeros, r_, asarray
from numpy.linalg import inv

from panel.iv.covariance import KERNEL_LOOKUP, HomoskedasticCovariance


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
    def __init__(self, center=False, kernel='bartlett', bandwidth=None):
        super(KernelWeightMatrix, self).__init__(center)
        self._bandwidth = bandwidth
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
        clusters = asarray(clusters).copy().squeeze()
        ind = argsort(clusters)
        ze = ze[ind]
        clusters = clusters[ind]

        locs = nonzero(r_[True, clusters[1:] != clusters[:-1], True])[0]
        st, en = locs[:-1], locs[1:]

        s = zeros((ninstr, ninstr))
        for sloc, eloc in zip(st, en):
            zec = ze[sloc:eloc].sum(axis=0)[None, :]
            s += zec.T @ zec

        return s / nobs

    @property
    def config(self):
        return {'center': self._center,
                'clusters': self._clusters}


class IVGMMCovariance(HomoskedasticCovariance):
    """
    Parameters
    ----------
    x : ndarray
        Model regressors (nobs by nvar)
    y : ndarray
        Series ,modeled (nobs by 1)
    z : ndarray
        Instruments used for endogensou regressors (nobs by ninstr)
    params : ndarray
        Estimated model parameters (nvar by 1)
    w : ndarray
        Weighting matrix used in GMM estimation
    """

    def __init__(self, x, y, z, params, w, cov_type='robust', **cov_config):
        super(IVGMMCovariance, self).__init__(x, y, z, params, False)
        self._cov_type = cov_type
        self._cov_config = cov_config
        self.w = w

    @property
    def cov(self):
        x, z, eps, w = self.x, self.z, self.eps, self.w
        nobs = x.shape[0]
        xpz = x.T @ z / nobs
        xpzw = xpz @ w
        xpzwzpx_inv = inv(xpzw @ xpz.T)

        if self._cov_type in ('robust', 'heteroskedastic'):
            score_cov_estimator = HeteroskedasticWeightMatrix
        elif self._cov_type in ('unadjusted', 'homoskedastic'):
            score_cov_estimator = HomoskedasticWeightMatrix
        elif self._cov_type == 'clustered':
            score_cov_estimator = OneWayClusteredWeightMatrix
        elif self._cov_type == 'kernel':
            score_cov_estimator = KernelWeightMatrix
        else:
            raise ValueError('Unknown cov_type')
        score_cov = score_cov_estimator(**self._cov_config)
        s = score_cov.weight_matrix(x, z, eps)

        return xpzwzpx_inv @ (xpzw @ s @ xpzw.T) @ xpzwzpx_inv / nobs

    @property
    def config(self):
        return {'debiased': self.debiased,
                'name': self.__class__.__name__}
