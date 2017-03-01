from __future__ import print_function, absolute_import, division

from numpy import asarray, unique
from numpy.linalg import inv

from panel.iv.covariance import (KERNEL_LOOKUP, HomoskedasticCovariance,
                                 _cov_cluster, _cov_kernel)


class HomoskedasticWeightMatrix(object):
    """
    Parameters
    ----------
    **weight_config
        Keywords to pass to weight matrix
    """

    def __init__(self, center=False, debiased=False):
        self._center = center
        self._debiased = debiased
        self._bandwidth = 0

    def weight_matrix(self, x, z, eps):
        nobs, nvar = x.shape
        # TODO: Determine if always remove this
        # mu = eps.mean(0) if self._center else 0
        mu = eps.mean(0)
        s2 = (eps - mu).T @ (eps - mu) / nobs
        w = s2 * z.T @ z / nobs
        w *= 1 if not self._debiased else nobs / (nobs - nvar)
        return w

    @property
    def config(self):
        return {'center': self._center,
                'debiased': self._debiased}


class HeteroskedasticWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, center=False, debiased=False):
        super(HeteroskedasticWeightMatrix, self).__init__(center, debiased)

    def weight_matrix(self, x, z, eps):
        nobs, nvar = x.shape
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        w = ze.T @ ze / nobs
        w *= 1 if not self._debiased else nobs / (nobs - nvar)
        return w


class KernelWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, center=False, kernel='bartlett', bandwidth=None, debiased=False):
        super(KernelWeightMatrix, self).__init__(center, debiased)
        self._bandwidth = bandwidth
        self._kernel = kernel
        self._kernels = KERNEL_LOOKUP

    def weight_matrix(self, x, z, eps):
        nobs, nvar = x.shape
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        # TODO: Fix this to allow optimal bw selection by default
        bw = self._bandwidth if self._bandwidth is not None else nobs - 2
        self._bandwidth = bw
        w = self._kernels[self._kernel](bw, nobs - 1)
        s = _cov_kernel(ze, w)
        s *= 1 if not self._debiased else nobs / (nobs - nvar)

        return s

    def _optimal_bandwidth(self, x, z, eps):
        # TODO: Implement this
        pass  # pragma: no cover

    @property
    def config(self):
        return {'center': self._center,
                'bandwidth': self._bandwidth,
                'kernel': self._kernel,
                'debiased': self._debiased}


class OneWayClusteredWeightMatrix(HomoskedasticWeightMatrix):
    def __init__(self, clusters, center=False, debiased=False):
        super(OneWayClusteredWeightMatrix, self).__init__(center, debiased)
        self._clusters = clusters

    def weight_matrix(self, x, z, eps):
        nobs, nvar = x.shape

        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        clusters = self._clusters
        if clusters.shape[0] != nobs:
            raise ValueError('clusters has the wrong nobs. Expected {0}, '
                             'got {1}'.format(nobs, clusters.shape[0]))
        clusters = asarray(clusters).copy().squeeze()

        s = _cov_cluster(ze, clusters)

        if self._debiased:
            num_clusters = len(unique(clusters))
            scale = (nobs - 1) / (nobs - nvar) * num_clusters / (num_clusters - 1)
            s *= scale

        return s

    @property
    def config(self):
        return {'center': self._center,
                'clusters': self._clusters,
                'debiased': self._debiased}


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

    def __init__(self, x, y, z, params, w, cov_type='robust', debiased=False,
                 **cov_config):
        super(IVGMMCovariance, self).__init__(x, y, z, params, debiased)
        self._cov_type = cov_type
        self._cov_config = cov_config
        self.w = w
        self._bandwidth = 0
        self._kernel = ''

    @property
    def cov(self):
        x, z, eps, w = self.x, self.z, self.eps, self.w
        nobs, nvar = x.shape
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
        score_cov = score_cov_estimator(debiased=self.debiased, **self._cov_config)
        s = score_cov.weight_matrix(x, z, eps)
        self._cov_config = score_cov.config

        c = xpzwzpx_inv @ (xpzw @ s @ xpzw.T) @ xpzwzpx_inv / nobs
        return (c + c.T) / 2

    @property
    def config(self):
        conf = {'type': self._cov_type,
                'debiased': self.debiased,
                'name': self.__class__.__name__}
        conf.update(self._cov_config)
        return conf
