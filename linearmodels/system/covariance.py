from numpy import eye, ones, sqrt, vstack, zeros
from numpy.linalg import inv

from linearmodels.system._utility import (blocked_diag_product, blocked_inner_prod,
                                          inv_matrix_sqrt)


class HomoskedasticCovariance(object):
    r"""
    Homoskedastic covariance estimation for system regression

    Parameters
    ----------
    x : list of ndarray
        List of regressor arrays (ndependent)
    eps : ndarray
        Model residuals, ndependent by nobs
    sigma : ndarray
        Covariance matrix estimator of eps
    gls : bool, optional
        Flag indicating to compute the GLS covariance estimator.  If False,
        assume OLS was used
    debiased : bool, optional
        Flag indicating to apply a small sample adjustment
    constraints : {None, LinearConstraint}, optional
        Constraints used in estimation, if any

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

    def __init__(self, x, eps, sigma, full_sigma, *, gls=False, debiased=False, constraints=None):
        self._eps = eps
        self._x = x
        self._nobs = eps.shape[0]
        self._k = len(x)
        self._sigma = sigma
        self._full_sigma = full_sigma
        self._gls = gls
        self._debiased = debiased
        self._constraints = constraints

    @property
    def sigma(self):
        """Error covariance"""
        return self._sigma

    def _adjustment(self):
        # Sigma is pre-debiased
        return 1.0

    def _mvreg_cov(self):
        x = self._x

        xeex = blocked_inner_prod(x, self._sigma)
        xpx = blocked_inner_prod(self._x, eye(len(x)))

        if self._constraints is None:
            xpxi = inv(xpx)
            cov = xpxi @ xeex @ xpxi
        else:
            cons = self._constraints
            xpx = cons.t.T @ xpx @ cons.t
            xpxi = inv(xpx)
            xeex = cons.t.T @ xeex @ cons.t
            cov = cons.t @ (xpxi @ xeex @ xpxi) @ cons.t.T

        cov = (cov + cov.T) / 2
        return cov

    def _gls_cov(self):
        x = self._x
        sigma = self._sigma
        sigma_inv = inv(sigma)

        xpx = blocked_inner_prod(x, sigma_inv)
        # Handles case where sigma_inv is not inverse of full_sigma
        xeex = blocked_inner_prod(x, sigma_inv @ self._full_sigma @ sigma_inv)
        if self._constraints is None:
            xpxi = inv(xpx)
            cov = xpxi @ xeex @ xpxi
        else:
            cons = self._constraints
            xpx = cons.t.T @ xpx @ cons.t
            xpxi = inv(xpx)
            xeex = cons.t.T @ xeex @ cons.t
            cov = cons.t @ (xpxi @ xeex @ xpxi) @ cons.t.T

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

    def __init__(self, x, eps, sigma, full_sigma, gls=False, debiased=False, constraints=None):
        super(HeteroskedasticCovariance, self).__init__(x, eps, sigma, full_sigma,
                                                        gls=gls,
                                                        debiased=debiased,
                                                        constraints=constraints)

    def _cov(self, gls):
        x = self._x
        eps = self._eps
        k = len(x)
        sigma = self.sigma
        weights = inv(sigma) if gls else eye(k)
        xpx = blocked_inner_prod(x, weights)

        weights = inv_matrix_sqrt(sigma) if gls else eye(k)
        bigx = blocked_diag_product(x, weights)
        nobs = eps.shape[0]
        e = eps.T.ravel()[:, None]
        bigxe = bigx * e
        m = bigx.shape[1]
        xeex = zeros((m, m))
        for i in range(nobs):
            xe = bigxe[i::nobs].sum(0)[None, :]
            xeex += xe.T @ xe

        if self._constraints is None:
            xpxi = inv(xpx)
            cov = xpxi @ xeex @ xpxi
        else:
            cons = self._constraints
            xpx = cons.t.T @ xpx @ cons.t
            xpxi = inv(xpx)
            xeex = cons.t.T @ xeex @ cons.t
            cov = cons.t @ (xpxi @ xeex @ xpxi) @ cons.t.T

        cov = (cov + cov.T) / 2
        return cov

    def _mvreg_cov(self):
        return self._cov(False)

    def _gls_cov(self):
        return self._cov(True)

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
