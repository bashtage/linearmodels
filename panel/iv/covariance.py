from __future__ import print_function, absolute_import, division
from __future__ import print_function, absolute_import, division

from numpy import ceil, where, argsort, r_, unique, zeros, arange, pi, sin, cos
from numpy.linalg import pinv, inv


def kernel_weight_bartlett(max_lag):
    """
    Kernel weights from a Bartlett kernel

    Parameters
    ----------
    max_lag : int
       Maximum lag to used in kernel

    Returns
    -------
    weights : ndarray
        Weight array  ordered by lag position (maxlag + 1)

    Notes
    -----
    .. math::

       w_i = 1 - i / (m + 1), \, i < m
    """
    return 1 - arange(max_lag + 1) / (max_lag + 1)


def kernel_weight_quadratic_spectral(max_lag):
    r"""
    Kernel weights from a quadratic-spectral kernel

    Parameters
    ----------
    max_lag : int
       Maximum lag to used in kernel

    Returns
    -------
    weights : ndarray
        Weight array  ordered by lag position (maxlag + 1)

    Notes
    -----
    .. math::

       z_i & = 6i\pi / 5                                        \\
       w_0 &  = 1                                                \\
       w_i &  = 3(\sin(z_i)/z_i - cos(z_i))/z_i^ 2, \, i \geq 1
    """
    w = 6 * pi * arange(max_lag + 1) / 5
    w[0] = 1
    w[1:] = 3 * (sin(w[1:]) / w[1:] - cos(w[1:])) / w[1:] ** 2
    return w


def kernel_weight_parzen(max_lag):
    r"""
    Kernel weights from a Parzen kernel

    Parameters
    ----------
    max_lag : int
       Maximum lag to used in kernel

    Returns
    -------
    weights : ndarray
        Weight array  ordered by lag position (maxlag + 1)

    Notes
    -----
    .. math::

       z_i & = i / (m+1)                    \\
       w_i &  = 1-6z_i^2+6z_i^3, z \leq 0.5 \\
       w_i &  = 2(1-z_i)^3, z > 0.5
    """
    z = arange(max_lag + 1) / (max_lag + 1)
    w = 1 - 6 * z ** 2 + 6 * z ** 3
    w[z > 0.5] = 2 * (1 - z[z > 0.5]) ** 3
    return w


KERNEL_LOOKUP = {'bartlett': kernel_weight_bartlett,
                 'newey-west': kernel_weight_bartlett,
                 'quadratic-spectral': kernel_weight_quadratic_spectral,
                 'qs': kernel_weight_quadratic_spectral,
                 'andrews': kernel_weight_quadratic_spectral,
                 'gallant': kernel_weight_parzen,
                 'parzen': kernel_weight_parzen}


class HomoskedasticCovariance(object):
    """
    Covariance estimation for homoskedastic data
    
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
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment
    """

    def __init__(self, x, y, z, params, debiased=False):
        self.x = x
        self.y = y
        self.z = z
        self.params = params
        self._debiased = debiased
        self.eps = y - x @ params
        self._pinvz = pinv(z)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        s2 = eps.T @ eps / nobs
        v = (x.T @ z) @ (pinv(z) @ x) / nobs

        return s2 * v

    @property
    def cov(self):
        """Covariance of estimated parameters"""

        x, z = self.x, self.z
        nobs, nvar = x.shape

        scale = nobs / (nobs - nvar) if self.config['debiased'] else 1
        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        vinv = inv(v)

        return scale * vinv @ self.s @ vinv / nobs

    @property
    def s2(self):
        """
        Estimated variance of residuals. Small-sample adjusted if debiased.
        """
        nobs, nvar = self.x.shape
        eps = self.eps
        denom = nobs - nvar if self.debiased else nobs
        return eps.T @ eps / denom

    @property
    def debiased(self):
        """Flag indicating whether to debias"""
        return self._debiased

    @property
    def config(self):
        return {'debiased': self.debiased,
                'name': self.__class__.__name__}


class KernelCovariance(HomoskedasticCovariance):
    """
    Kernel weighted (HAC) covariance estimation

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
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment
    kernel : str
        Kernel name. Supported kernels are: 

        * 'bartlett', 'newey-west' - Triangular kernel 
        * 'qs', 'quadratic-spectral', 'andrews' - Quadratic spectral kernel
        * 'parzen', 'gallant' - Parzen's kernel;
          
    bandwidth : {int, None}
        Non-negative bandwidth to use with kernel. If None, automatic
        bandwidth selection is used.

    """

    def __init__(self, x, y, z, params, debiased=False, kernel='bartlett',
                 bandwidth=None):
        super(KernelCovariance, self).__init__(x, y, z, params, debiased)
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._kernels = KERNEL_LOOKUP

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape

        kernel = self.config['kernel']
        # TODO: Bandwidth selection method
        bw = self.config['bw']
        if bw is None:
            if kernel in ('newey-west', 'bartlett'):
                bw = ceil(20 * (nobs / 100) ** (2 / 9))
            elif kernel in ('andrews', 'quadratic-spectral', 'qs'):
                bw = ceil(20 * (nobs / 100) ** (2 / 25))
            elif kernel in ('parzen', 'gallant'):
                bw = ceil(20 * (nobs / 100) ** (4 / 25))
            else:
                raise ValueError('Unknown kernel {0}'.format(kernel))
        bw = int(bw)
        w = self._kernels[kernel](bw)

        pinvz = self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e

        for i in range(bw):
            s += 2 * w[i + 1] * xhat_e[i + 1:].T @ xhat_e[:-(i + 1)]
        s /= nobs

        return s

    @property
    def config(self):
        return {'debiased': self.debiased,
                'bw': self._bandwidth,
                'kernel': self._kernel,
                'name': self.__class__.__name__}


class HeteroskedasticCovariance(HomoskedasticCovariance):
    """
    Covariance estimation for heteroskedastic data

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
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment

    """

    def __init__(self, x, y, z, params, debiased=False):
        super(HeteroskedasticCovariance, self).__init__(x, y, z, params, debiased)

    @property
    def s(self):
        x, z, eps = self.x, self.z, self.eps
        nobs, nvar = x.shape
        pinvz = self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e / nobs
        return s


class OneWayClusteredCovariance(HomoskedasticCovariance):
    """
    Covariance estimation for clustered data

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
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment
    clusters : ndarray, optional
        Cluster group assignment.  If not provided, uses clusters of 1
    """

    def __init__(self, x, y, z, params, debiased=False, clusters=None):
        super(OneWayClusteredCovariance, self).__init__(x, y, z, params,
                                                        debiased)
        self._clusters = clusters

    @property
    def s(self):
        """One-wasy clustered estimator of score covariance"""
        x, z, eps = self.x, self.z, self.eps
        pinvz = self._pinvz
        xhat_e = z @ (pinvz @ x) * eps

        nobs, nvar = x.shape
        clusters = self._clusters
        clusters = arange(nobs) if clusters is None else clusters
        self._clusters = clusters
        clusters = clusters.copy().squeeze()
        num_clusters = len(unique(clusters))

        sort_args = argsort(clusters)
        clusters = clusters[sort_args]
        locs = where(r_[True, clusters[:-1] != clusters[1:], True])[0]
        xhat_e = xhat_e[sort_args]

        s = zeros((nvar, nvar))
        for i in range(num_clusters):
            st, en = locs[i], locs[i + 1]
            xhat_e_bar = xhat_e[st:en].sum(axis=0)[:, None]
            s += xhat_e_bar @ xhat_e_bar.T

        s *= num_clusters / (num_clusters - 1) / nobs

        return s

    @property
    def config(self):
        return {'debiased': self.debiased,
                'clusters': self._clusters,
                'name': self.__class__.__name__}


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

    def __init__(self, x, y, z, params, w, **cov_config):
        super(IVGMMCovariance, self).__init__(x, y, z, params, False)
        self.w = w

    @property
    def cov(self):
        x, z, eps, w = self.x, self.z, self.eps, self.w
        nobs = x.shape[0]
        xpz = x.T @ z / nobs
        xpzw = xpz @ w
        xpzwzpx_inv = inv(xpzw @ xpz.T)

        # TODO: Need to use cov options here
        # TODO: Simple "robust" s for now
        # TODO: HAC, "robust" (s=w^-1), cluter, homoskedastic
        ze = z * eps
        s = ze.T @ ze / nobs

        return xpzwzpx_inv @ (xpzw @ s @ xpzw.T) @ xpzwzpx_inv / nobs

    @property
    def config(self):
        return {'debiased': self.debiased,
                'name': self.__class__.__name__}
