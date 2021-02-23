from typing import List, Optional, Union, cast

from numpy import asarray, empty, eye, ndarray, ones, sqrt, vstack, zeros
from numpy.linalg import inv

from linearmodels.asset_pricing.covariance import _HACMixin
from linearmodels.iv.covariance import cov_cluster
from linearmodels.panel.covariance import cluster_union
from linearmodels.shared.covariance import group_debias_coefficient
from linearmodels.shared.utility import AttrDict
from linearmodels.system._utility import (
    LinearConstraint,
    blocked_cross_prod,
    blocked_diag_product,
    blocked_inner_prod,
)
from linearmodels.typing import NDArray

CLUSTERS_FORMAT = """\
clusters must be an ndarray with as shape (nobs, ncluster) where ncluster is the \
number of clustering variables.  Clustering is only supported in 1 or 2 dimensions.
"""


class HomoskedasticCovariance(object):
    r"""
    Homoskedastic covariance estimation for system regression

    Parameters
    ----------
    x : List[ndarray]
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

    def __init__(
        self,
        x: List[ndarray],
        eps: NDArray,
        sigma: NDArray,
        full_sigma: NDArray,
        *,
        gls: bool = False,
        debiased: bool = False,
        constraints: Optional[LinearConstraint] = None,
    ) -> None:
        self._eps = eps
        self._x = x
        self._nobs = eps.shape[0]
        self._k = len(x)
        self._sigma = sigma
        self._full_sigma = full_sigma
        self._gls = gls
        self._debiased = debiased
        self._constraints = constraints
        self._name = "Homoskedastic (Unadjusted) Covariance"
        self._str_extra = AttrDict(Debiased=self._debiased, GLS=self._gls)
        self._cov_config = AttrDict(debiased=self._debiased)

    def __str__(self) -> str:
        out = self._name
        extra: List[str] = []
        for key in self._str_extra:
            extra.append(": ".join([str(key), str(self._str_extra[key])]))
        if extra:
            out += " (" + ", ".join(extra) + ")"
        return out

    def __repr__(self) -> str:
        out = self.__str__()
        return out + ", id: {0}".format(hex(id(self)))

    @property
    def sigma(self) -> NDArray:
        """Error covariance"""
        return self._sigma

    def _adjustment(self) -> Union[float, ndarray]:
        # Sigma is pre-debiased
        return 1.0

    def _mvreg_cov(self) -> NDArray:
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

    def _gls_cov(self) -> NDArray:
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
    def cov(self) -> NDArray:
        """Parameter covariance"""
        adj = self._adjustment()
        if self._gls:
            return cast(ndarray, adj * self._gls_cov())
        else:
            return cast(ndarray, adj * self._mvreg_cov())

    @property
    def cov_config(self) -> AttrDict:
        """Optional configuration information used in covariance"""
        return self._cov_config


class HeteroskedasticCovariance(HomoskedasticCovariance):
    r"""
    Heteroskedastic covariance estimation for system regression

    Parameters
    ----------
    x : List[ndarray]
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
    constraints : {None, LinearConstraint}, optional
        Constraints used in estimation, if any

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

    def __init__(
        self,
        x: List[ndarray],
        eps: NDArray,
        sigma: NDArray,
        full_sigma: NDArray,
        *,
        gls: bool = False,
        debiased: bool = False,
        constraints: Optional[LinearConstraint] = None,
    ) -> None:

        super(HeteroskedasticCovariance, self).__init__(
            x,
            eps,
            sigma,
            full_sigma,
            gls=gls,
            debiased=debiased,
            constraints=constraints,
        )
        self._name = "Heteroskedastic (Robust) Covariance"

        k = len(x)
        nobs = eps.shape[0]

        if gls:
            weights = inv(sigma)
            bigx = blocked_diag_product(x, weights)
            e = eps.T.ravel()[:, None]
            bigxe = bigx * e
            m = bigx.shape[1]
            xe = zeros((nobs, m))
            for i in range(nobs):
                xe[i, :] = bigxe[i::nobs].sum(0)[None, :]
        else:
            # Do not require blocking when not using GLS
            k_tot = sum(map(lambda a: a.shape[1], x))
            xe = empty((nobs, k_tot))
            loc = 0
            for i in range(k):
                offset = x[i].shape[1]
                xe[:, loc : loc + offset] = x[i] * eps[:, i : i + 1]
                loc += offset

        self._moments = xe

    def _xeex(self) -> NDArray:
        nobs = self._moments.shape[0]
        return self._moments.T @ self._moments / nobs

    def _cov(self, gls: bool) -> NDArray:
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
            assert self._constraints is not None
            cons = self._constraints
            xpx = cons.t.T @ xpx @ cons.t
            xpxi = inv(xpx)
            xeex = cons.t.T @ xeex @ cons.t
            cov = cons.t @ (xpxi @ xeex @ xpxi) @ cons.t.T

        cov = (cov + cov.T) / 2
        return cov / nobs

    def _mvreg_cov(self) -> NDArray:
        return self._cov(False)

    def _gls_cov(self) -> NDArray:
        return self._cov(True)

    def _adjustment(self) -> Union[float, ndarray]:
        if not self._debiased:
            return 1.0
        ks = [s.shape[1] for s in self._x]
        nobs = self._x[0].shape[0]
        adj = []
        for k in ks:
            adj.append(nobs / (nobs - k) * ones((k, 1)))
        adj_arr = vstack(adj)
        adj_arr = sqrt(adj_arr)
        # TODO: Check Type
        return adj_arr @ adj_arr.T


class KernelCovariance(HeteroskedasticCovariance, _HACMixin):
    r"""
    Kernel (HAC) covariance estimation for system regression

    Parameters
    ----------
    x : List[ndarray]
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

    def __init__(
        self,
        x: List[ndarray],
        eps: NDArray,
        sigma: NDArray,
        full_sigma: NDArray,
        *,
        gls: bool = False,
        debiased: bool = False,
        constraints: Optional[LinearConstraint] = None,
        kernel: str = "bartlett",
        bandwidth: Optional[float] = None,
    ):
        _HACMixin.__init__(self, kernel, bandwidth)
        super(KernelCovariance, self).__init__(
            x,
            eps,
            sigma,
            full_sigma,
            gls=gls,
            debiased=debiased,
            constraints=constraints,
        )

        self._check_kernel(kernel)
        self._check_bandwidth(bandwidth)
        self._name = "Kernel (HAC) Covariance"
        self._str_extra["Kernel"] = kernel
        self._cov_config["kernel"] = kernel

    def _xeex(self) -> NDArray:
        return self._kernel_cov(self._moments)

    @property
    def cov_config(self) -> AttrDict:
        """Optional configuration information used in covariance"""
        out = AttrDict([(k, v) for k, v in self._cov_config.items()])
        out["bandwidth"] = self.bandwidth
        return out


class ClusteredCovariance(HeteroskedasticCovariance):
    r"""
    Heteroskedastic covariance estimation for system regression

    Parameters
    ----------
    x : List[ndarray]
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
    constraints : {None, LinearConstraint}, optional
        Constraints used in estimation, if any
    clusters : ndarray, default None
        Optional array of cluster id.  Must be integer valued, and have shape
        (nobs, ncluster) where ncluster is 1 or 2.
    group_debias : bool, default False
        Flag indicating whether to debias by the number of groups.

    Notes
    -----
    If GLS is used, the covariance is estimated by

    .. math::

        (X'\Omega^{-1}X)^{-1}\tilde{S}_{\mathcal{G}}(X'\Omega^{-1}X)^{-1}

    where X is a block diagonal matrix of exogenous variables and where
    :math:`\tilde{S}_{\mathcal{G}}` is a clustered estimator of the model
    scores based on the model residuals and the weighted X matrix
    :math:`\Omega^{-1/2}X`.

    When GLS is not used, the covariance is estimated by

    .. math::

        (X'X)^{-1}\hat{S}_{\mathcal{G}}(X'X)^{-1}

    where :math:`\hat{S}` is a clustered estimator of the covariance of the
    model scores.

    See Also
    --------
    linearmodels.shared.covariance.cov_cluster
    linearmodels.shared.covariance.group_debias_coefficient
    """

    def __init__(
        self,
        x: List[ndarray],
        eps: NDArray,
        sigma: NDArray,
        full_sigma: NDArray,
        *,
        gls: bool = False,
        debiased: bool = False,
        constraints: Optional[LinearConstraint] = None,
        clusters: Optional[NDArray] = None,
        group_debias: bool = False,
    ) -> None:
        super().__init__(
            x,
            eps,
            sigma,
            full_sigma,
            gls=gls,
            debiased=debiased,
            constraints=constraints,
        )
        self._group_debias = group_debias
        self._nclusters: List[int] = []
        self._clusters = self._check_clusters(clusters)
        self._str_extra["Number of Grouping Variables"] = self._clusters.shape[1]
        if self._clusters.shape[1] > 0:
            num_cl = [f"{nc} (Variable {i})" for i, nc in enumerate(self._nclusters)]
            self._str_extra["Number of Groups"] = " and ".join(num_cl)
        self._str_extra["Group Debias"] = self._group_debias

    def _check_clusters(self, clusters: Optional[NDArray]) -> NDArray:
        """Check cluster dimension and ensure ndarray"""
        if clusters is None:
            return empty((self._eps.size, 0))
        _clusters = asarray(clusters)

        if _clusters.ndim not in (1, 2):
            raise ValueError(CLUSTERS_FORMAT, ValueError)
        elif _clusters.ndim == 1:
            _clusters = _clusters[:, None]
        shape = _clusters.shape
        if shape[0] != self._eps.shape[0] or not 1 <= shape[1] <= 2:
            raise ValueError(CLUSTERS_FORMAT, ValueError)
        from pandas import DataFrame

        df = DataFrame(_clusters)
        nunique = df.nunique()
        if _clusters.shape[1] == 2:
            both = df.groupby([0, 1]).ngroups
            if both == nunique.max():
                raise ValueError(
                    "clusters must be non-nested. You must drop nested "
                    "the nested cluster before computing the clustered"
                    "covariance."
                )
        self._nclusters = list(nunique)
        return _clusters

    def _xeex(self) -> NDArray:
        if self._clusters.shape[1] == 0:
            # Heteroskedastic but not clustered
            return super()._xeex()
        elif self._clusters.shape[1] == 1:
            s = cov_cluster(self._moments, self._clusters[:, 0])
            if self._group_debias:
                s *= group_debias_coefficient(self._clusters[:, 0])
            return s

        else:
            xeex0 = cov_cluster(self._moments, self._clusters[:, 0])
            xeex1 = cov_cluster(self._moments, self._clusters[:, 1])

            clusters01 = cluster_union(self._clusters)
            xeex01 = cov_cluster(self._moments, clusters01)

            if self._group_debias:
                xeex0 *= group_debias_coefficient(self._clusters[:, 0])
                xeex1 *= group_debias_coefficient(self._clusters[:, 1])
                xeex01 *= group_debias_coefficient(clusters01)

            return xeex0 + xeex1 - xeex01

    @property
    def cov_config(self) -> AttrDict:
        """Optional configuration information used in covariance"""
        out = AttrDict([(k, v) for k, v in self._cov_config.items()])
        out["clusters"] = self._clusters
        out["group_debias"] = self._group_debias
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

    def __init__(
        self,
        x: List[ndarray],
        z: List[ndarray],
        eps: NDArray,
        w: NDArray,
        *,
        sigma: Optional[ndarray] = None,
        debiased: bool = False,
        constraints: Optional[LinearConstraint] = None,
    ) -> None:
        self._x = x
        self._z = z
        self._eps = eps
        self._sigma = sigma
        self._w = w
        self._debiased = debiased
        self._constraints = constraints
        self._name = "GMM Homoskedastic (Unadjusted) Covariance"
        self._cov_config = AttrDict(debiased=self._debiased)

    def __str__(self) -> str:
        out = self._name
        return out

    def __repr__(self) -> str:
        out = self.__str__()
        return out + ", id: {0}".format(hex(id(self)))

    @property
    def cov(self) -> NDArray:
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
            cov = (
                cons.t
                @ xpz_wi_zpxi
                @ xpz_wi_omega_wi_zpx
                @ xpz_wi_zpxi
                @ cons.t.T
                / nobs
            )

        cov = (cov + cov.T) / 2
        return adj * cov

    def _omega(self) -> NDArray:
        z = self._z
        nobs = z[0].shape[0]
        sigma = self._sigma
        assert sigma is not None
        omega = blocked_inner_prod(z, sigma)
        omega /= nobs

        return omega

    def _adjustment(self) -> Union[float, ndarray]:
        if not self._debiased:
            return 1.0
        k = [s.shape[1] for s in self._x]
        nobs = self._x[0].shape[0]
        adj = []
        for i in range(len(k)):
            adj.append(nobs / (nobs - k[i]) * ones((k[i], 1)))
        adj_arr = vstack(adj)
        adj_arr = sqrt(adj_arr)
        return adj_arr @ adj_arr.T

    @property
    def cov_config(self) -> AttrDict:
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

    def __init__(
        self,
        x: List[ndarray],
        z: List[ndarray],
        eps: NDArray,
        w: NDArray,
        *,
        sigma: Optional[ndarray] = None,
        debiased: bool = False,
        constraints: Optional[LinearConstraint] = None,
    ) -> None:
        super().__init__(
            x, z, eps, w, sigma=sigma, debiased=debiased, constraints=constraints
        )
        self._name = "GMM Heteroskedastic (Robust) Covariance"

        k = len(z)
        k_total = sum(map(lambda a: a.shape[1], z))
        nobs = z[0].shape[0]
        loc = 0
        ze = empty((nobs, k_total))
        for i in range(k):
            kz = z[i].shape[1]
            ze[:, loc : loc + kz] = z[i] * eps[:, [i]]
            loc += kz
        self._moments = ze

    def _omega(self) -> NDArray:
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

    def __init__(
        self,
        x: List[ndarray],
        z: List[ndarray],
        eps: NDArray,
        w: NDArray,
        *,
        sigma: Optional[ndarray] = None,
        debiased: bool = False,
        constraints: Optional[LinearConstraint] = None,
        kernel: str = "bartlett",
        bandwidth: Optional[float] = None,
    ) -> None:
        _HACMixin.__init__(self, kernel, bandwidth)
        super().__init__(
            x, z, eps, w, sigma=sigma, debiased=debiased, constraints=constraints
        )
        self._name = "GMM Kernel (HAC) Covariance"
        self._check_bandwidth(bandwidth)
        self._check_kernel(kernel)
        self._cov_config["kernel"] = kernel

    def _omega(self) -> NDArray:
        return self._kernel_cov(self._moments)

    @property
    def cov_config(self) -> AttrDict:
        """Optional configuration information used in covariance"""
        out = AttrDict([(k, v) for k, v in self._cov_config.items()])
        out["bandwidth"] = self.bandwidth
        return out
