from numpy import eye, zeros_like, ones, vstack, sqrt, zeros
from numpy.linalg import inv

from linearmodels.system._utility import blocked_inner_prod, blocked_diag_product


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
    debiased : bool
        Flag indicating to apply a small sample adjustment

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

    def __init__(self, x, eps, sigma, *, gls=False, debiased=False):
        self._eps = eps
        self._x = x
        self._nobs = eps.shape[0]
        self._k = len(x)
        self._sigma = sigma
        self._gls = gls
        self._debiased = debiased

    @property
    def sigma(self):
        """Error covariance"""
        return self._sigma

    def _adjustment(self):
        if not self._debiased:
            return 1.0
        k = list(map(lambda s: s.shape[1], self._x))
        nobs = self._x[0].shape[0]
        adj = []
        for i in range(len(k)):
            adj.append(nobs / (nobs - k[i]) * ones((k[i], 1)))
        adj = vstack(adj)
        adj = sqrt(adj)
        return adj @ adj.T

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
        """Parameter covariance"""
        adj = self._adjustment()
        if self._gls:
            return adj * self._gls_cov()
        else:
            return adj * self._mvreg_cov()


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
    debiased : bool
        Flag indicating to apply a small sample adjustment

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

    # TODO: Groupby time period before computing xeex
    def __init__(self, x, eps, sigma, gls=False, debiased=False):
        super(HeteroskedasticCovariance, self).__init__(x, eps, sigma,
                                                        gls=gls,
                                                        debiased=debiased)

    def _cov(self, gls):
        x = self._x
        eps = self._eps
        k = len(x)
        sigma = self.sigma
        inv_sigma = inv(sigma)
        weights = inv_sigma if gls else eye(k)
        xpx = blocked_inner_prod(x, weights)
        xpxi = inv(xpx)

        bigx = blocked_diag_product(x, weights)
        nobs = eps.shape[0]
        e = eps.T.ravel()[:, None]
        bigxe = bigx * e
        m = bigx.shape[1]
        xeex = zeros((m, m))
        for i in range(nobs):
            xe = bigxe[i:k * nobs: nobs].sum(0)[None, :]
            xeex += xe.T @ xe

        cov = xpxi @ xeex @ xpxi
        cov = (cov + cov.T) / 2
        return cov

    def _mvreg_cov(self):
        return self._cov(False)

    def _gls_cov(self):
        return self._cov(True)