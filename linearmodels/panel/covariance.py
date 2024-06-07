from __future__ import annotations

from functools import cached_property
from typing import Any, Union

import numpy as np
from numpy.linalg import inv
import pandas
from pandas import DataFrame, MultiIndex

from linearmodels.iv.covariance import (
    CLUSTER_ERR,
    KERNEL_LOOKUP,
    cov_cluster,
    cov_kernel,
    kernel_optimal_bandwidth,
)
from linearmodels.shared.covariance import cluster_union, group_debias_coefficient
from linearmodels.shared.typed_getters import (
    get_array_like,
    get_bool,
    get_float,
    get_string,
)
import linearmodels.typing.data

__all__ = [
    "HomoskedasticCovariance",
    "HeteroskedasticCovariance",
    "ClusteredCovariance",
    "DriscollKraay",
    "CovarianceManager",
    "ACCovariance",
    "FamaMacBethCovariance",
    "setup_covariance_estimator",
]


class HomoskedasticCovariance:
    r"""
    Homoskedastic covariance estimation

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool
        Flag indicating whether to debias the estimator
    extra_df : int
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased

    Notes
    -----
    The estimator of the covariance is

    .. math:: s^2\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = X'X

    and

    .. math::

        s^2 = (n-df)^{-1} \hat{\epsilon}'\hat{\epsilon}

    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``.
    """

    ALLOWED_KWARGS: tuple[str, ...] = tuple()
    DEFAULT_KERNEL = "newey-west"

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.IntArray | None,
        time_ids: linearmodels.typing.data.IntArray | None,
        *,
        debiased: bool = False,
        extra_df: int = 0,
    ):
        self._y = y
        self._x = x
        self._params = params
        self._entity_ids = entity_ids
        self._time_ids = time_ids
        self._debiased = debiased
        self._extra_df = extra_df
        self._nobs, self._nvar = x.shape
        self._nobs_eff = self._nobs - extra_df
        if debiased:
            self._nobs_eff -= self._nvar
        self._scale = self._nobs / self._nobs_eff
        self._name = "Unadjusted"

    @property
    def name(self) -> str:
        """Covariance estimator name"""
        return self._name

    @property
    def eps(self) -> linearmodels.typing.data.Float64Array:
        """Model residuals"""
        return self._y - self._x @ self._params

    @property
    def s2(self) -> float:
        """Error variance"""
        eps = self.eps
        return self._scale * float(np.squeeze(eps.T @ eps)) / self._nobs

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance"""
        x = self._x
        out = self.s2 * inv(x.T @ x)
        return (out + out.T) / 2

    def deferred_cov(self) -> linearmodels.typing.data.Float64Array:
        """Covariance calculation deferred until executed"""
        return self.cov


class HeteroskedasticCovariance(HomoskedasticCovariance):
    r"""
    Covariance estimation using White estimator

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool
        Flag indicating whether to debias the estimator
    extra_df : int
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased

    Notes
    -----
    The estimator of the covariance is

    .. math::

        n^{-1}\hat{\Sigma}_{xx}^{-1}\hat{S}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = n^{-1}X'X

    and

    .. math::

        \hat{S} = (n-df)^{-1} \sum_{i=1}^n \hat{\epsilon}_i^2 x_i'x_i

    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``.
    """

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.IntArray,
        time_ids: linearmodels.typing.data.IntArray,
        *,
        debiased: bool = False,
        extra_df: int = 0,
    ) -> None:
        super().__init__(
            y, x, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
        )
        self._name = "Robust"

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance"""
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps
        xe = x * eps
        xeex = self._scale * xe.T @ xe / nobs

        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


class ClusteredCovariance(HomoskedasticCovariance):
    r"""
    One-way (Rogers) or two-way clustered covariance estimation

    Parameters
    ----------
    y : ndarray
        nobs by 1 stacked array of dependent
    x : ndarray
        nobs by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool
        Flag indicating whether to debias the estimator
    extra_df : int
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased
    clusters : ndarray
        nobs by 1 or nobs by 2 array of cluster group ids
    group_debias : bool
        Flag indicating whether to apply small-number of groups adjustment.

    Notes
    -----
    The estimator of the covariance is

    .. math::

        \hat{\Sigma}_{xx}^{-1}\hat{S}_{\mathcal{G}}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = X'X

    and :math:`\hat{S}_{\mathcal{G}}` is a one- or two-way cluster covariance
    of the scores.  Two-way clustering is implemented by summing up the two
    one-way cluster covariances and then subtracting the one-way clustering
    covariance computed using the group formed from the intersection of the
    two groups.

    Two small sample adjustment are available.  ``debias=True`` will account
    for regressors in the main model. ``group_debias=True`` will provide a
    small sample adjustment for the number of clusters of the form

    .. math ::

      (g / (g- 1)) ((n - 1) / n)

    where g is the number of distinct groups and n is the number of
    observations.
    """

    ALLOWED_KWARGS = ("clusters", "group_debias")

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.IntArray,
        time_ids: linearmodels.typing.data.IntArray,
        *,
        debiased: bool = False,
        extra_df: int = 0,
        clusters: linearmodels.typing.data.ArrayLike | None = None,
        group_debias: bool = False,
    ) -> None:
        super().__init__(
            y, x, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
        )
        if clusters is None:
            clusters = np.arange(self._x.shape[0])
        clusters = np.asarray(clusters).squeeze()
        assert clusters is not None
        self._group_debias = bool(group_debias)
        dim1 = 1 if clusters.ndim == 1 else clusters.shape[1]
        if clusters.ndim > 2 or dim1 > 2:
            raise ValueError("Only 1 or 2-way clustering supported.")
        nobs = y.shape[0]
        if clusters.shape[0] != nobs:
            raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))
        self._clusters = clusters
        self._name = "Clustered"

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance"""
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)

        eps = self.eps
        xe = x * eps
        if self._clusters.ndim == 1:
            xeex = cov_cluster(xe, self._clusters)
            if self._group_debias:
                xeex *= group_debias_coefficient(self._clusters)

        else:
            clusters0 = self._clusters[:, 0]
            clusters1 = self._clusters[:, 1]
            xeex0 = cov_cluster(xe, clusters0)
            xeex1 = cov_cluster(xe, clusters1)

            clusters01 = cluster_union(self._clusters)
            xeex01 = cov_cluster(xe, clusters01)

            if self._group_debias:
                xeex0 *= group_debias_coefficient(clusters0)
                xeex1 *= group_debias_coefficient(clusters1)
                xeex01 *= group_debias_coefficient(clusters01)

            xeex = xeex0 + xeex1 - xeex01

        xeex *= self._scale
        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


class DriscollKraay(HomoskedasticCovariance):
    r"""
    Driscoll-Kraay heteroskedasticity-autocorrelation robust covariance estimation

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool
        Flag indicating whether to debias the estimator
    extra_df : int
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased.
    kernel : str
        Name of one of the supported kernels. If None, uses the Newey-West
        kernel.
    bandwidth : int
        Non-negative integer to use as bandwidth.  If not provided a rule-of-
        thumb value is used.

    Notes
    -----
    Supported kernels:

    * "bartlett", "newey-west" - Bartlett's kernel
    * "quadratic-spectral", "qs", "andrews" - Quadratic-Spectral Kernel
    * "parzen", "gallant" - Parzen kernel

    Bandwidth is set to the common value for the Bartlett kernel if not
    provided.

    The estimator of the covariance is

    .. math::

        n^{-1}\hat{\Sigma}_{xx}^{-1}\hat{S}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = n^{-1}X'X

    and

    .. math::
      \xi_t & = \sum_{i=1}^{n_t} \epsilon_i x_{i} \\
      \hat{S}_0 & = \sum_{i=1}^{t} \xi'_t \xi_t \\
      \hat{S}_j & = \sum_{i=1}^{t-j} \xi'_t \xi_{t+j} + \xi'_{t+j} \xi_t  \\
      \hat{S}   & = (n-df)^{-1} \sum_{j=0}^{bw} K(j, bw) \hat{S}_j

    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``. :math:`K(i, bw)` is the kernel weighting function.
    """

    ALLOWED_KWARGS = ("kernel", "bandwidth")
    # TODO: Test

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.IntArray,
        time_ids: linearmodels.typing.data.IntArray,
        *,
        debiased: bool = False,
        extra_df: int = 0,
        kernel: str | None = None,
        bandwidth: float | None = None,
    ) -> None:
        super().__init__(
            y, x, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
        )
        self._name = "Driscoll-Kraay"
        self._kernel = kernel if kernel is not None else self.DEFAULT_KERNEL
        self._bandwidth = bandwidth

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance"""
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps

        xe = x * eps
        assert self._time_ids is not None
        xe_df = DataFrame(xe, index=self._time_ids.squeeze())
        xe_df = xe_df.groupby(level=0).sum()
        xe_df.sort_index(inplace=True)
        xe_nobs = xe_df.shape[0]
        bw = self._bandwidth
        if self._bandwidth is None:
            bw = float(np.floor(4 * (xe_nobs / 100) ** (2 / 9)))
        assert bw is not None
        w = KERNEL_LOOKUP[self._kernel](bw, xe_nobs - 1)
        xeex = cov_kernel(np.asarray(xe_df), w) * (xe_nobs / nobs)
        xeex *= self._scale

        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


class ACCovariance(HomoskedasticCovariance):
    r"""
    Autocorrelation robust covariance estimation

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        variables by 1 array of estimated model parameters
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids
    debiased : bool
        Flag indicating whether to debias the estimator
    extra_df : int
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects.  Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased
    kernel : str
        Name of one of the supported kernels. If None, uses the Newey-West
        kernel.
    bandwidth : int
        Non-negative integer to use as bandwidth.  If not provided a rule-of-
        thumb value is used.

    Notes
    -----
    Estimator is robust to autocorrelation but not cross-sectional correlation.

    Supported kernels:

    * "bartlett", "newey-west" - Bartlett's kernel
    * "quadratic-spectral", "qs", "andrews" - Quadratic-Spectral Kernel
    * "parzen", "gallant" - Parzen kernel

    Bandwidth is set to the common value for the Bartlett kernel if not
    provided.

    The estimator of the covariance is

    .. math::

        n^{-1}\hat{\Sigma}_{xx}^{-1}\hat{S}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = n^{-1}X'X

    and

    .. math::

      \xi_t & = \epsilon_{it} x_{it} \\
      \hat{S} & = n / (N(n-df))  \sum_{i=1}^N S_i \\
      \hat{S}_i & = \sum_{j=0}^{bw} K(j, bw) \hat{S}_{ij} \\
      \hat{S}_{i0} & = \sum_{t=1}^{T} \xi'_{it} \xi_{it} \\
      \hat{S}_{ij} & = \sum_{t=1}^{T-j} \xi'_{it} \xi_{it+j} + \xi'_{it+j} \xi_{it}


    where df is ``extra_df`` and n-df is replace by n-df-k if ``debiased`` is
    ``True``. :math:`K(i, bw)` is the kernel weighting function.
    """

    ALLOWED_KWARGS = ("kernel", "bandwidth")
    # TODO: Docstring

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.IntArray,
        time_ids: linearmodels.typing.data.IntArray,
        *,
        debiased: bool = False,
        extra_df: int = 0,
        kernel: str | None = None,
        bandwidth: float | None = None,
    ) -> None:
        super().__init__(
            y, x, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
        )
        self._name = "Autocorrelation Rob. Cov."
        self._kernel = kernel if kernel is not None else self.DEFAULT_KERNEL
        self._bandwidth = bandwidth

    def _single_cov(
        self, xe: linearmodels.typing.data.Float64Array, bw: float
    ) -> linearmodels.typing.data.Float64Array:
        nobs = xe.shape[0]
        w = KERNEL_LOOKUP[self._kernel](bw, nobs - 1)
        return cov_kernel(xe, w)

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance"""
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps
        assert self._time_ids is not None
        time_ids = np.unique(self._time_ids.squeeze())
        nperiods = len(time_ids)
        bw = self._bandwidth
        if self._bandwidth is None:
            bw = float(np.floor(4 * (nperiods / 100) ** (2 / 9)))
        assert bw is not None

        xe = x * eps
        assert self._entity_ids is not None
        index = [self._entity_ids.squeeze(), self._time_ids.squeeze()]
        xe_df = DataFrame(xe, index=index)
        xe_df = xe_df.sort_index(level=[0, 1])
        xe_index = xe_df.index
        assert isinstance(xe_index, MultiIndex)
        entities = xe_index.levels[0]
        nentity = len(entities)
        xeex = np.zeros((xe_df.shape[1], xe_df.shape[1]))
        for entity in entities:
            _xe = np.asarray(xe_df.loc[entity])
            _bw = min(_xe.shape[0] - 1.0, bw)
            xeex += self._single_cov(_xe, _bw)
        xeex /= nentity
        xeex *= self._scale

        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


CovarianceEstimator = Union[
    HomoskedasticCovariance,
    HeteroskedasticCovariance,
    ClusteredCovariance,
    DriscollKraay,
    ACCovariance,
]
CovarianceEstimatorType = Union[
    type[HomoskedasticCovariance],
    type[HeteroskedasticCovariance],
    type[ClusteredCovariance],
    type[DriscollKraay],
    type[ACCovariance],
]


class CovarianceManager:
    COVARIANCE_ESTIMATORS: dict[str, CovarianceEstimatorType] = {
        "unadjusted": HomoskedasticCovariance,
        "conventional": HomoskedasticCovariance,
        "homoskedastic": HomoskedasticCovariance,
        "robust": HeteroskedasticCovariance,
        "heteroskedastic": HeteroskedasticCovariance,
        "clustered": ClusteredCovariance,
        "driscoll-kraay": DriscollKraay,
        "dk": DriscollKraay,
        "kernel": DriscollKraay,
        "ac": ACCovariance,
        "autocorrelated": ACCovariance,
    }

    def __init__(
        self, estimator: str, *cov_estimators: CovarianceEstimatorType
    ) -> None:
        self._estimator = estimator
        self._supported = cov_estimators

    def __getitem__(self, item: str) -> CovarianceEstimatorType:
        if item not in self.COVARIANCE_ESTIMATORS:
            raise KeyError("Unknown covariance estimator type.")
        cov_est = self.COVARIANCE_ESTIMATORS[item]
        if cov_est not in self._supported:
            raise ValueError(
                "Requested covariance estimator is not supported "
                "for the {}.".format(self._estimator)
            )
        return cov_est


class FamaMacBethCovariance(HomoskedasticCovariance):
    """
    HAC estimator for Fama-MacBeth estimator

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of dependent
    x : ndarray
        (entity x time) by variables stacked array of exogenous
    params : ndarray
        (variables by 1) array of estimated model parameters
    all_params : ndarray
        (nobs by variables) array of all estimated model parameters
    debiased : bool
        Flag indicating whether to debias the estimator.
    bandwidth : int
        Non-negative integer to use as bandwidth.  Set to 0 to disable
        autocorrelation robustness. If not provided a rule-of- thumb
        value is used.
    kernel : str
        Name of one of the supported kernels. If None, uses the Newey-West
        kernel.


    Notes
    -----
    Covariance is a Kernel covariance of all estimated parameters.
    """

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        all_params: pandas.DataFrame,
        *,
        debiased: bool = False,
        bandwidth: float | None = None,
        kernel: str | None = None,
    ) -> None:
        super().__init__(y, x, params, None, None, debiased=debiased)
        self._all_params = all_params
        cov_type = "Standard " if bandwidth == 0 else "Kernel "
        self._name = f"Fama-MacBeth {cov_type}Cov"
        self._bandwidth = bandwidth
        self._kernel = kernel if kernel is not None else self.DEFAULT_KERNEL

    @cached_property
    def bandwidth(self) -> float:
        """Estimator bandwidth"""
        if self._bandwidth is None:
            all_params = np.asarray(self._all_params)
            e = all_params - self._params.T
            e = e[np.all(np.isfinite(e), 1)]
            stde = np.sum(e / e.std(0)[None, :], 1)
            self._bandwidth = kernel_optimal_bandwidth(stde, self._kernel)
        assert self._bandwidth is not None
        return self._bandwidth

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance"""
        e = np.asarray(self._all_params) - self._params.T
        e = e[np.all(np.isfinite(e), 1)]
        nobs = e.shape[0]

        bw = self.bandwidth
        assert self._kernel is not None
        w = KERNEL_LOOKUP[self._kernel](bw, nobs - 1)
        cov = cov_kernel(e, w)
        return cov / (nobs - int(bool(self._debiased)))

    @property
    def all_params(self) -> DataFrame:
        """
        The set of parameters estimated for each of the time periods

        Returns
        -------
        DataFrame
            The parameters (nobs, nparam).
        """
        return self._all_params


def setup_covariance_estimator(
    cov_estimators: CovarianceManager,
    cov_type: str,
    y: linearmodels.typing.data.Float64Array,
    x: linearmodels.typing.data.Float64Array,
    params: linearmodels.typing.data.Float64Array,
    entity_ids: linearmodels.typing.data.IntArray,
    time_ids: linearmodels.typing.data.IntArray,
    *,
    debiased: bool = False,
    extra_df: int = 0,
    **cov_config: Any,
) -> HomoskedasticCovariance:
    estimator = cov_estimators[cov_type]
    unknown_kwargs = [
        str(key) for key in cov_config if str(key) not in estimator.ALLOWED_KWARGS
    ]
    if unknown_kwargs:
        if estimator.ALLOWED_KWARGS:
            allowed = ", ".join(estimator.ALLOWED_KWARGS)
            kwarg_err = f"only supports the keyword arguments: {allowed}"
        else:
            kwarg_err = "does not support any keyword arguments"
        msg = (
            f"Covariance estimator {estimator.__name__} {kwarg_err}. Unknown keyword "
            f"arguments were passed to the estimator. The unknown keyword argument(s) "
            f"are: {', '.join(unknown_kwargs)} "
        )
        raise ValueError(msg)
    kernel = get_string(cov_config, "kernel")
    bandwidth = get_float(cov_config, "bandwidth")
    group_debias = get_bool(cov_config, "group_debias")
    clusters = get_array_like(cov_config, "clusters")

    if estimator is HomoskedasticCovariance:
        return HomoskedasticCovariance(
            y, x, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
        )
    elif estimator is HeteroskedasticCovariance:
        return HeteroskedasticCovariance(
            y, x, params, entity_ids, time_ids, debiased=debiased, extra_df=extra_df
        )
    elif estimator is ClusteredCovariance:
        return ClusteredCovariance(
            y,
            x,
            params,
            entity_ids,
            time_ids,
            debiased=debiased,
            extra_df=extra_df,
            clusters=clusters,
            group_debias=group_debias,
        )
    elif estimator is DriscollKraay:
        return DriscollKraay(
            y,
            x,
            params,
            entity_ids,
            time_ids,
            debiased=debiased,
            extra_df=extra_df,
            kernel=kernel,
            bandwidth=bandwidth,
        )
    else:  # ACCovariance:
        return ACCovariance(
            y,
            x,
            params,
            entity_ids,
            time_ids,
            debiased=debiased,
            extra_df=extra_df,
            kernel=kernel,
            bandwidth=bandwidth,
        )
