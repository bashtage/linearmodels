from numpy import empty, eye, ones, sqrt, vstack, zeros
from numpy.linalg import inv

from linearmodels.asset_pricing.covariance import _HACMixin
from linearmodels.system._utility import (blocked_cross_prod, blocked_diag_product,
                                          blocked_inner_prod)
from linearmodels.utility import AttrDict


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
        self._name = 'Homoskedastic (Unadjusted) Covariance'
        self._str_extra = AttrDict(Debiased=self._debiased, GLS=self._gls)
        self._cov_config = AttrDict(debiased=self._debiased)

    def __str__(self):
        out = self._name
        extra = []
        for key in self._str_extra:
            extra.append(': '.join([key, str(self._str_extra[key])]))
        if extra:
            out += ' (' + ', '.join(extra) + ')'
        return out

    def __repr__(self):
        out = self.__str__()
        return out + ', id: {0}'.format(hex(id(self)))

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

    @property
    def cov_config(self):
        """Optional configuration information used in covariance"""
        return self._cov_config


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

    where :math:`\hat{S}` is a estimator of the covariance of the model scores.
    """

    def __init__(self, x, eps, sigma, full_sigma, gls=False, debiased=False, constraints=None):
        super(HeteroskedasticCovariance, self).__init__(x, eps, sigma, full_sigma,
                                                        gls=gls,
                                                        debiased=debiased,
                                                        constraints=constraints)
        self._name = 'Heteroskedastic (Robust) Covariance'

        k = len(x)
        weights = inv(sigma) if gls else eye(k)
        bigx = blocked_diag_product(x, weights)
        nobs = eps.shape[0]
        e = eps.T.ravel()[:, None]
        bigxe = bigx * e
        m = bigx.shape[1]
        xe = zeros((nobs, m))
        for i in range(nobs):
            xe[i, :] = bigxe[i::nobs].sum(0)[None, :]
        self._moments = xe

    def _xeex(self):
        nobs = self._moments.shape[0]
        return self._moments.T @ self._moments / nobs

    def _cov(self, gls):
        x = self._x
        nobs = x[0].shape[0]
        k = len(x)
        sigma = self.sigma
        weights = inv(sigma) if gls else eye(k)
        xpx = blocked_inner_prod(x, weights) / nobs
        xeex = self._xeex()

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
        return cov / nobs

    def _mvreg_cov(self):
        return self._cov(False)

    def _gls_cov(self):
        return self._cov(True)

    def _adjustment(self):
        if not self._debiased:
            return 1.0
        ks = list(map(lambda s: s.shape[1], self._x))
        nobs = self._x[0].shape[0]
        adj = []
        for k in ks:
            adj.append(nobs / (nobs - k) * ones((k, 1)))
        adj = vstack(adj)
        adj = sqrt(adj)
        return adj @ adj.T


class KernelCovariance(HeteroskedasticCovariance, _HACMixin):
    r"""
    Kernel (HAC) covariance estimation for system regression

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
    kernel : str, optional
        Name of kernel to use.  Supported kernels include:

          * 'bartlett', 'newey-west' : Bartlett's kernel
          * 'parzen', 'gallant' : Parzen's kernel
          * 'qs', 'quadratic-spectral', 'andrews' : Quadratic spectral kernel

    bandwidth : float, optional
        Bandwidth to use for the kernel.  If not provided the optimal
        bandwidth will be estimated.

    Notes
    -----
    If GLS is used, the covariance is estimated by

    .. math::

        (X'\Omega^{-1}X)^{-1}\tilde{S}(X'\Omega^{-1}X)^{-1}

    where X is a block diagonal matrix of exogenous variables and where
    :math:`\tilde{S}` is a estimator of the covariance of the model scores
    based on the model residuals and the weighted X matrix :math:`\Omega^{-1/2}X`.

    When GLS is not used, the covariance is estimated by

    .. math::

        (X'X)^{-1}\hat{S}(X'X)^{-1}

    where :math:`\hat{S}` is a estimator of the covariance of the model scores.

    See Also
    --------
    linearmodels.iv.covariance.kernel_weight_bartlett,
    linearmodels.iv.covariance.kernel_weight_parzen,
    linearmodels.iv.covariance.kernel_weight_quadratic_spectral
    """

    def __init__(self, x, eps, sigma, full_sigma, *, gls=False, debiased=False, constraints=None,
                 kernel='bartlett', bandwidth=None):
        super(KernelCovariance, self).__init__(x, eps, sigma, full_sigma, gls=gls,
                                               debiased=debiased,
                                               constraints=constraints)

        self._check_kernel(kernel)
        self._check_bandwidth(bandwidth)
        self._name = 'Kernel (HAC) Covariance'
        self._str_extra['Kernel'] = kernel
        self._cov_config['kernel'] = kernel

    def _xeex(self):
        return self._kernel_cov(self._moments)

    @property
    def cov_config(self):
        """Optional configuration information used in covariance"""
        out = AttrDict([(k, v) for k, v in self._cov_config.items()])
        out['bandwidth'] = self.bandwidth
        return out


class GMMHomoskedasticCovariance(object):
    r"""
    Covariance estimator for IV system estimation with homoskedastic data

    Parameters
    ----------
    x : List[ndarray]
        List containing the model regressors for each equation in the system
    z : List[ndarray]
        List containing the model instruments for each equation in the system
    eps : ndarray
        nobs by neq array of residuals where each column corresponds an
        equation in the system
    w : ndarray
        Weighting matrix used in estimation
    sigma : ndarray, optional
        Residual covariance used in estimation
    constraints : {None, LinearConstraint}, optional
        Constraints used in estimation, if any

    Notes
    -----
    The covariance is estimated by

    .. math::

      (X'ZW^{-1}Z'X)^{-1}(X'ZW^{-1}\Omega W^{-1}Z'X)(X'ZW^{-1}Z'X)^{-1}

    where :math:`\Omega = W = Z'(\Sigma \otimes I_N)Z` where m is the number of
    moments in the system
    """

    def __init__(self, x, z, eps, w, *, sigma=None, debiased=False, constraints=None):
        self._x = x
        self._z = z
        self._eps = eps
        self._sigma = sigma
        self._w = w
        self._debiased = debiased
        self._constraints = constraints
        self._name = 'GMM Homoskedastic (Unadjusted) Covariance'
        self._cov_config = AttrDict(debiased=self._debiased)

    def __str__(self):
        out = self._name
        return out

    def __repr__(self):
        out = self.__str__()
        return out + ', id: {0}'.format(hex(id(self)))

    @property
    def cov(self):
        """Parameter covariance"""
        x, z = self._x, self._z
        k = len(x)
        nobs = x[0].shape[0]
        xpz = blocked_cross_prod(x, z, eye(k))
        xpz /= nobs
        wi = inv(self._w)
        xpz_wi_zpx = xpz @ wi @ xpz.T

        omega = self._omega()
        xpz_wi_omega_wi_zpx = xpz @ wi @ omega @ wi @ xpz.T
        adj = self._adjustment()
        if self._constraints is None:
            xpz_wi_zpxi = inv(xpz_wi_zpx)
            cov = xpz_wi_zpxi @ xpz_wi_omega_wi_zpx @ xpz_wi_zpxi / nobs
        else:
            cons = self._constraints
            xpz_wi_zpx = cons.t.T @ xpz_wi_zpx @ cons.t
            xpz_wi_zpxi = inv(xpz_wi_zpx)
            xpz_wi_omega_wi_zpx = cons.t.T @ xpz_wi_omega_wi_zpx @ cons.t
            cov = cons.t @ xpz_wi_zpxi @ xpz_wi_omega_wi_zpx @ xpz_wi_zpxi @ cons.t.T / nobs

        cov = (cov + cov.T) / 2
        return adj * cov

    def _omega(self):
        z = self._z
        nobs = z[0].shape[0]
        sigma = self._sigma
        omega = blocked_inner_prod(z, sigma)
        omega /= nobs

        return omega

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

    @property
    def cov_config(self):
        """Optional configuration information used in covariance"""
        return self._cov_config


class GMMHeteroskedasticCovariance(GMMHomoskedasticCovariance):
    r"""
    Covariance estimator for IV system estimation with homoskedastic data

    Parameters
    ----------
    x : List[ndarray]
        List containing the model regressors for each equation in the system
    z : List[ndarray]
        List containing the model instruments for each equation in the system
    eps : ndarray
        nobs by neq array of residuals where each column corresponds an
        equation in the system
    w : ndarray
        Weighting matrix used in estimation
    sigma : ndarray, optional
        Residual covariance used in estimation
    constraints : {None, LinearConstraint}, optional
        Constraints used in estimation, if any

    Notes
    -----
    The covariance is estimated by

    .. math::

      (X'ZW^{-1}Z'X)^{-1}(X'ZW^{-1}\Omega W^{-1}Z'X)(X'ZW^{-1}Z'X)^{-1}

    where :math:`\Omega` is the covariance of the moment conditions.
    """

    def __init__(self, x, z, eps, w, *, sigma=None, debiased=False, constraints=None):
        super().__init__(x, z, eps, w, sigma=sigma, debiased=debiased, constraints=constraints)
        self._name = 'GMM Heteroskedastic (Robust) Covariance'

        k = len(z)
        k_total = sum(map(lambda a: a.shape[1], z))
        nobs = z[0].shape[0]
        loc = 0
        ze = empty((nobs, k_total))
        for i in range(k):
            kz = z[i].shape[1]
            ze[:, loc:loc + kz] = z[i] * eps[:, [i]]
            loc += kz
        self._moments = ze

    def _omega(self):
        z = self._z
        nobs = z[0].shape[0]
        omega = self._moments.T @ self._moments / nobs

        return omega


class GMMKernelCovariance(GMMHeteroskedasticCovariance, _HACMixin):
    r"""
    Covariance estimator for IV system estimation with homoskedastic data

    Parameters
    ----------
    x : List[ndarray]
        List containing the model regressors for each equation in the system
    z : List[ndarray]
        List containing the model instruments for each equation in the system
    eps : ndarray
        nobs by neq array of residuals where each column corresponds an
        equation in the system
    w : ndarray
        Weighting matrix used in estimation
    sigma : ndarray, optional
        Residual covariance used in estimation
    constraints : {None, LinearConstraint}, optional
        Constraints used in estimation, if any
    kernel : str, optional
        Name of kernel to use.  Supported kernels include:

          * 'bartlett', 'newey-west' : Bartlett's kernel
          * 'parzen', 'gallant' : Parzen's kernel
          * 'qs', 'quadratic-spectral', 'andrews' : Quadratic spectral kernel

    bandwidth : float, optional
        Bandwidth to use for the kernel.  If not provided the optimal
        bandwidth will be estimated.

    Notes
    -----
    The covariance is estimated by

    .. math::

      (X'ZW^{-1}Z'X)^{-1}(X'ZW^{-1}\Omega W^{-1}Z'X)(X'ZW^{-1}Z'X)^{-1}

    where :math:`\Omega` is the covariance of the moment conditions.
    """

    def __init__(self, x, z, eps, w, *, sigma=None, debiased=False, constraints=None,
                 kernel='bartlett', bandwidth=None):
        super().__init__(x, z, eps, w, sigma=sigma, debiased=debiased, constraints=constraints)
        self._name = 'GMM Kernel (HAC) Covariance'
        self._check_bandwidth(bandwidth)
        self._check_kernel(kernel)
        self._cov_config['kernel'] = kernel

    def _omega(self):
        return self._kernel_cov(self._moments)

    @property
    def cov_config(self):
        """Optional configuration information used in covariance"""
        out = AttrDict([(k, v) for k, v in self._cov_config.items()])
        out['bandwidth'] = self.bandwidth
        return out
