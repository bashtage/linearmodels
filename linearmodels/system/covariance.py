from numpy import eye, zeros_like
from numpy.linalg import inv

from linearmodels.system._utility import blocked_inner_prod


class HomoskedasticCovariance(object):
    r"""
    Homoskedastic covariance estimation for system regression

    Parameters
    ----------
    x : list of ndarray
        ndependent element list of regressor
    eps : ndarray
        Model residuals, ndependent by nobs
    sigma : ndarray
        Covariance matrix estimator of eps
    gls : bool
        Flag indicating to compute the GLS covariance estimator.  If False,
        assume OLS was used

    Notes
    -----
    If GLS is used, the covariance is estimated by

    .. math::

        (X'\Omega^{-1}X)^{-1}

    where X is a block diagonal matrix of exogenous variables. When GLS is
    not used, the covariance is estimated by

    .. math::

        (X'X)^{-1}(X'\Omega X)(X'X)^{-1}

    """

    def __init__(self, x, eps, sigma, *, gls=False):
        self._eps = eps
        self._x = x
        self._nobs = eps.shape[0]
        self._k = len(x)
        self._sigma = sigma
        self._gls = gls

    @property
    def sigma(self):
        return self._sigma

    def _mvreg_cov(self):
        x = self._x
        k = len(x)
        sigma = self.sigma
        xeex = blocked_inner_prod(x, sigma)
        xpxi = zeros_like(xeex)
        loc = 0
        for i in range(k):
            ki = x[i].shape[1]
            xpxi[loc:loc + ki, loc: loc + ki] = inv(x[i].T @ x[i])
            loc += ki
        cov = xpxi @ xeex @ xpxi
        cov = (cov + cov.T) / 2
        return cov

    def _gls_cov(self):
        x = self._x
        sigma = self.sigma

        cov = blocked_inner_prod(x, inv(sigma))
        cov = (cov + cov.T) / 2
        cov = inv(cov)
        cov = (cov + cov.T) / 2

        return cov

    @property
    def cov(self):
        if self._gls:
            return self._gls_cov()
        else:
            return self._mvreg_cov()


class HeteroskedasticCovariance(HomoskedasticCovariance):
    r"""
    Heteroskedastic covariance estimation for system regression

    Parameters
    ----------
    x : list of ndarray
        ndependent element list of regressor
    eps : ndarray
        Model residuals, ndependent by nobs
    sigma : ndarray
        Covariance matrix estimator of eps
    gls : bool
        Flag indicating to compute the GLS covariance estimator.  If False,
        assume OLS was used

    Notes
    -----
    If GLS is used, the covariance is estimated by

    .. math::

        (X'\Omega^{-1}X)^{-1}\tilde{S}(X'\Omega^{-1}X)^{-1}

    where X is a block diagonal matrix of exogenous variables and where
    :math:`\tilde{S}` is a estimator of the model scores based on the model
    residuals and the weighted X matrix :math:`\Omega^{-1/2}X`.

    When GLS is not used, the covariance is estimated by

    .. math::

        (X'X)^{-1}\hat{S}(X'X)^{-1}

    where :math:`\hat{S}` is a estimator of the model scores.

    """

    def __init__(self, x, eps, sigma, gls=False):
        super(HeteroskedasticCovariance, self).__init__(x, eps, sigma, gls=gls)

    def _cov(self, gls):
        x = self._x
        eps = self._eps
        k = len(x)
        sigma = self.sigma
        inv_sigma = inv(sigma)
        weights = inv_sigma if gls else eye(k)
        xpx = blocked_inner_prod(x, weights)
        xpxi = inv(xpx)

        xe = []
        for i in range(k):
            xe.append(x[i] * eps[:, [i]])
        xeex = blocked_inner_prod(xe, weights)

        cov = xpxi @ xeex @ xpxi
        cov = (cov + cov.T) / 2
        return cov

    def _mvreg_cov(self):
        return self._cov(False)

    def _gls_cov(self):
        return self._cov(True)
