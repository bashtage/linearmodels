"""Covariance estimators for panel models."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Any, ClassVar, Union

import numpy as np
from numpy.linalg import inv
from pandas import DataFrame, MultiIndex

from linearmodels.iv.covariance import (
    CLUSTER_ERR,
    KERNEL_LOOKUP,
    _min_nobs_automatic_bandwidth,
    cov_cluster,
    cov_kernel,
    kernel_optimal_bandwidth,
)
from linearmodels.shared.covariance import cluster_union, group_debias_coefficient
from linearmodels.shared.exceptions import (
    check_cov_config,
    lookup_option,
    lower_option,
    unknown_option_error,
)
from linearmodels.shared.typed_getters import (
    get_array_like,
    get_bool,
    get_float,
    get_string,
)
import linearmodels.typing.data

__all__ = [
    "ACCovariance",
    "ClusteredCovariance",
    "CovarianceManager",
    "DriscollKraay",
    "FamaMacBethCovariance",
    "HeteroskedasticCovariance",
    "HomoskedasticCovariance",
    "setup_covariance_estimator",
]


class HomoskedasticCovariance:
    r"""
    Homoskedastic covariance estimation.

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of the dependent variable.
    x : ndarray
        (entity x time) by variables stacked array of the exogenous variables.
    params : ndarray
        Variables by 1 array of estimated model parameters.
    entity_ids : ndarray or None
        (entity x time) by 1 stacked array of entity ids. Not used by this
        estimator, so it can be None.
    time_ids : ndarray or None
        (entity x time) by 1 stacked array of time ids. Not used by this
        estimator, so it can be None.
    debiased : bool, optional
        Flag indicating whether to debias the estimator. The default is False.
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects. Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased. The
        default is 0.

    Raises
    ------
    ValueError
        If the model has no residual degrees of freedom: the number of
        observations less ``extra_df``, and less the number of regressors if
        ``debiased`` is True, is not positive.

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

    where df is ``extra_df`` and n-df is replaced by n-df-k if ``debiased`` is
    ``True``.
    """

    ALLOWED_KWARGS: tuple[str, ...] = ()
    DEFAULT_KERNEL = "newey-west"

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.AnyIntArray | None,
        time_ids: linearmodels.typing.data.AnyIntArray | None,
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
        if self._nobs_eff <= 0:
            regressors = f"{self._nvar} regressors and " if debiased else ""
            raise ValueError(
                "The covariance of the model cannot be estimated since the model has "
                f"no residual degrees of freedom. It has {self._nobs} observations "
                f"and uses {regressors}{extra_df} degrees of freedom for the effects."
            )
        self._scale = self._nobs / self._nobs_eff
        self._name = "Unadjusted"

    @property
    def name(self) -> str:
        """Covariance estimator name."""
        return self._name

    @property
    def eps(self) -> linearmodels.typing.data.Float64Array:
        """Model residuals."""
        return self._y - self._x @ self._params

    @property
    def s2(self) -> float:
        """Error variance."""
        eps = self.eps
        return self._scale * float(np.squeeze(eps.T @ eps)) / self._nobs

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance."""
        x = self._x
        out = self.s2 * inv(x.T @ x)
        return (out + out.T) / 2

    def deferred_cov(self) -> linearmodels.typing.data.Float64Array:
        """
        Covariance calculation deferred until executed.

        Returns
        -------
        ndarray
            The estimated covariance of the parameters.
        """
        return self.cov


class HeteroskedasticCovariance(HomoskedasticCovariance):
    r"""
    Covariance estimation using White estimator.

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of the dependent variable.
    x : ndarray
        (entity x time) by variables stacked array of the exogenous variables.
    params : ndarray
        Variables by 1 array of estimated model parameters.
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids.
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids.
    debiased : bool, optional
        Flag indicating whether to debias the estimator. The default is False.
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects. Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased. The
        default is 0.

    Raises
    ------
    ValueError
        If the model has no residual degrees of freedom: the number of
        observations less ``extra_df``, and less the number of regressors if
        ``debiased`` is True, is not positive.

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

    where df is ``extra_df`` and n-df is replaced by n-df-k if ``debiased`` is
    ``True``.
    """

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.AnyIntArray,
        time_ids: linearmodels.typing.data.AnyIntArray,
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
        """Estimated covariance."""
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
    One-way (Rogers) or two-way clustered covariance estimation.

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of the dependent variable.
    x : ndarray
        (entity x time) by variables stacked array of the exogenous variables.
    params : ndarray
        Variables by 1 array of estimated model parameters.
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids.
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids.
    debiased : bool, optional
        Flag indicating whether to debias the estimator. The default is False.
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects. Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased. The
        default is 0.
    clusters : array_like, optional
        Array of cluster group ids with nobs rows and 1 or 2 columns. If not
        provided, every observation is its own cluster.
    group_debias : bool, optional
        Flag indicating whether to apply small-number of groups adjustment.
        The default is False.

    Raises
    ------
    ValueError
        If the model has no residual degrees of freedom: the number of
        observations less ``extra_df``, and less the number of regressors if
        ``debiased`` is True, is not positive. It is also raised if
        `clusters` has more than 2 columns or does not have nobs rows.

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

    Two small sample adjustments are available. ``debiased=True`` will account
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
        entity_ids: linearmodels.typing.data.AnyIntArray,
        time_ids: linearmodels.typing.data.AnyIntArray,
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
        """Estimated covariance."""
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
    Driscoll-Kraay heteroskedasticity-autocorrelation robust covariance estimation.

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of the dependent variable.
    x : ndarray
        (entity x time) by variables stacked array of the exogenous variables.
    params : ndarray
        Variables by 1 array of estimated model parameters.
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids.
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids.
    debiased : bool, optional
        Flag indicating whether to debias the estimator. The default is False.
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects. Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased. The
        default is 0.
    kernel : str, optional
        Name of the kernel. If not provided, "newey-west", which is the Bartlett
        kernel, is used. The names are not case sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    bandwidth : float, optional
        Non-negative bandwidth to use. It must be below the number of time
        periods. The Bartlett and Parzen kernels use its integer part. If not
        provided a rule-of-thumb value is used.

    Raises
    ------
    ValueError
        If the bandwidth, or the default bandwidth if none is given, is not
        below the number of time periods. It is also raised if the model has
        no residual degrees of freedom: the number of observations less
        ``extra_df``, and less the number of regressors if ``debiased`` is
        True, is not positive.
    UnknownOptionError
        If ``kernel`` is not one of the supported names. It is both a
        ``ValueError`` and a ``KeyError``.

    Notes
    -----
    Bandwidth is set to the common value for the Bartlett kernel if not
    provided, :math:`\lfloor 4 (T / 100)^{2/9} \rfloor` where :math:`T` is the
    number of time periods.

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

    where df is ``extra_df`` and n-df is replaced by n-df-k if ``debiased`` is
    ``True``. :math:`K(i, bw)` is the kernel weighting function.

    The sums :math:`\xi_t` of the scores of the :math:`T` time periods have
    autocovariances up to lag :math:`T-1`, so the bandwidth must be below
    :math:`T`. The bandwidth is checked when the estimator is created, so that
    the ``fit`` method of a model raises the error.

    .. versionchanged:: 7.1

       A bandwidth that is not below the number of time periods raises a
       ``ValueError``. Earlier versions raised "Length of w is larger than the
       number of elements in z" with the Bartlett and Parzen kernels, when the
       covariance was first used. The quadratic spectral kernel, which has no
       truncation, returned a covariance without an error. With a large bandwidth
       the weights of all of the lags are close to one, and the sum of the scores
       over all of the periods is zero in a regression, so that the covariance
       was close to zero.

    .. versionchanged:: 7.1

       A ``kernel`` that is not one of the supported names raises an
       ``UnknownOptionError`` when the estimator is created, so that the ``fit``
       method of a model raises. Earlier versions raised it when the covariance
       was first used, for example when ``std_errors`` of the results was read,
       so that ``fit`` returned results that raised when the covariance was used.
    """

    ALLOWED_KWARGS = ("kernel", "bandwidth")
    # TODO: Test

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.AnyIntArray,
        time_ids: linearmodels.typing.data.AnyIntArray,
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
        # Checked here so that fit raises, instead of the first use of the covariance
        self._kernel_weights = lookup_option(KERNEL_LOOKUP, self._kernel, "kernel")
        self._bw = self._select_bandwidth(np.unique(time_ids).shape[0])

    def _select_bandwidth(self, nperiods: int) -> float:
        """
        Select the bandwidth of the kernel, which has to be below the number of periods.

        Parameters
        ----------
        nperiods : int
            The number of time periods.

        Returns
        -------
        float
            The bandwidth that was given or, if there is none, the rule-of-thumb value.

        Raises
        ------
        ValueError
            If the bandwidth is not below the number of time periods.
        """
        bw = self._bandwidth
        if bw is None:
            bw = float(np.floor(4 * (nperiods / 100) ** (2 / 9)))
        if bw >= nperiods:
            which = "bandwidth" if self._bandwidth is not None else "default bandwidth"
            raise ValueError(
                f"The {which} of the Driscoll-Kraay covariance, {bw:g}, must be below "
                f"the number of time periods, {nperiods}. The sums of the scores of "
                f"{nperiods} periods have autocovariances up to lag {nperiods - 1}. "
                "Use a smaller bandwidth."
            )
        return bw

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance."""
        x = self._x
        nobs = x.shape[0]
        xpxi = inv(x.T @ x / nobs)
        eps = self.eps

        xe = x * eps
        assert self._time_ids is not None
        xe_df = DataFrame(xe, index=self._time_ids.squeeze())
        xe_df = xe_df.groupby(level=0).sum()
        xe_df = xe_df.sort_index()
        xe_nobs = xe_df.shape[0]
        w = self._kernel_weights(self._bw, xe_nobs - 1)
        xeex = cov_kernel(np.asarray(xe_df), w) * (xe_nobs / nobs)
        xeex *= self._scale

        out = (xpxi @ xeex @ xpxi) / nobs
        return (out + out.T) / 2


class ACCovariance(HomoskedasticCovariance):
    r"""
    Autocorrelation robust covariance estimation.

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of the dependent variable.
    x : ndarray
        (entity x time) by variables stacked array of the exogenous variables.
    params : ndarray
        Variables by 1 array of estimated model parameters.
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids.
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids.
    debiased : bool, optional
        Flag indicating whether to debias the estimator. The default is False.
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x, e.g., fixed effects. Covariance estimators are always
        adjusted for extra_df irrespective of the setting of debiased. The
        default is 0.
    kernel : str, optional
        Name of the kernel. If not provided, "newey-west", which is the Bartlett
        kernel, is used. The names are not case sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    bandwidth : float, optional
        Non-negative bandwidth to use. The Bartlett and Parzen kernels use its
        integer part. If not provided a rule-of-thumb value is used. It can be
        larger than the number of observations of an entity.

    Raises
    ------
    ValueError
        If the model has no residual degrees of freedom: the number of
        observations less ``extra_df``, and less the number of regressors if
        ``debiased`` is True, is not positive.
    UnknownOptionError
        If ``kernel`` is not one of the supported names. It is both a
        ``ValueError`` and a ``KeyError``.

    Notes
    -----
    Estimator is robust to autocorrelation but not cross-sectional correlation.

    Bandwidth is set to the common value for the Bartlett kernel if not
    provided, :math:`\lfloor 4 (T / 100)^{2/9} \rfloor` where :math:`T` is the
    number of time periods.

    The estimator of the covariance is

    .. math::

        n^{-1}\hat{\Sigma}_{xx}^{-1}\hat{S}\hat{\Sigma}_{xx}^{-1}

    where

    .. math::

        \hat{\Sigma}_{xx} = n^{-1}X'X

    and

    .. math::

      \xi_{it} & = \epsilon_{it} x_{it} \\
      \hat{S} & = (n-df)^{-1} \sum_{i=1}^N S_i \\
      \hat{S}_i & = \sum_{j=0}^{T_i-1} K(j, bw) \hat{S}_{ij} \\
      \hat{S}_{i0} & = \sum_{t=1}^{T_i} \xi'_{it} \xi_{it} \\
      \hat{S}_{ij} & = \sum_{t=1}^{T_i-j} \xi'_{it} \xi_{it+j} + \xi'_{it+j} \xi_{it}

    where :math:`T_i` is the number of observations of entity :math:`i`, df is
    ``extra_df`` and n-df is replaced by n-df-k if ``debiased`` is ``True``.
    :math:`K(j, bw)` is the kernel weighting function, which is 1 for the lag 0.
    The Bartlett and Parzen kernels are 0 for the lags above ``bandwidth``.

    **Lags**

    The lag between two observations of an entity is the difference between their
    positions in the observations of the entity, which are in the order of time. It is
    not the difference between their time periods. If a period is missing between two
    observations of an entity, they are one lag apart. With the Bartlett kernel the
    covariance is that of the cluster-HAC option of statsmodels
    (``cov_type="hac-panel"``), which also counts positions. plm (``vcovNW``) and
    sandwich (``vcovPL``) pair the observations that are :math:`j` periods apart, so
    that their covariances are different if an entity has missing periods between its
    observations: in a panel where about a quarter of the periods are missing the
    standard errors differed by about 3%. They are the same if no entity has a gap.
    See :ref:`panel-ac-lags`.

    **Entities with few observations**

    An entity has the lags :math:`0,\ldots,T_i-1`, and its sum is over these lags.
    The weights are those of the bandwidth that was chosen, :math:`K(j, bw)`, also
    for an entity with fewer observations than the bandwidth. The bandwidth is not
    reduced to :math:`T_i-1` for such an entity, and the weights of the lags that the
    entity does not have are not used. This is how plm, sandwich and statsmodels
    compute the covariance.

    .. versionchanged:: 7.1

       The entities are not weighted by their numbers of observations: each
       entity contributes the sum of the products of its scores. Earlier versions
       divided the sum of each entity by its number of observations and averaged
       over the entities, which is the same for a balanced panel. For an unbalanced
       panel the covariance, e.g., with a bandwidth of 0, was not the heteroskedasticity
       robust covariance, and differed from the covariance of other packages.

    .. versionchanged:: 7.1

       An entity with :math:`T_i` observations, where :math:`T_i \leq bw`, keeps the
       weights of the bandwidth. Earlier versions replaced the bandwidth of the
       entity by :math:`T_i-1`. This changed the Bartlett weights of the entity from
       :math:`1-j/(bw+1)` to :math:`1-j/T_i`, the Parzen weights in the same way,
       and the bandwidth of the quadratic spectral kernel. The covariance changes if
       any entity has fewer observations than the bandwidth, which is more likely
       in a panel where the entities are short or the bandwidth is large.

    .. versionchanged:: 7.1

       A ``kernel`` that is not one of the supported names raises an
       ``UnknownOptionError`` when the estimator is created, so that the ``fit``
       method of a model raises. Earlier versions raised it when the covariance
       was first used, for example when ``std_errors`` of the results was read,
       so that ``fit`` returned results that raised when the covariance was used.
    """

    ALLOWED_KWARGS = ("kernel", "bandwidth")
    # TODO: Docstring

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        entity_ids: linearmodels.typing.data.AnyIntArray,
        time_ids: linearmodels.typing.data.AnyIntArray,
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
        # Checked here so that fit raises, instead of the first use of the covariance
        self._kernel_weights = lookup_option(KERNEL_LOOKUP, self._kernel, "kernel")

    def _single_cov(
        self, xe: linearmodels.typing.data.Float64Array, bw: float
    ) -> linearmodels.typing.data.Float64Array:
        nobs = xe.shape[0]
        w = self._kernel_weights(bw, nobs - 1)
        # The entity has the lags 0, ..., nobs - 1 only. They have the weights of
        # the bandwidth that was chosen, also if the entity is shorter than the
        # bandwidth, and so the weights of the other lags are not used
        return cov_kernel(xe, w[:nobs])

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance."""
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
        xeex = np.zeros((xe_df.shape[1], xe_df.shape[1]))
        for entity in entities:
            _xe = np.asarray(xe_df.loc[entity])
            # The sum of the products of the scores of the entity, which the kernel
            # covariance divides by the number of observations of the entity
            xeex += self._single_cov(_xe, bw) * _xe.shape[0]
        xeex /= nobs
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
    """
    Look up the covariance estimators that a model supports.

    Parameters
    ----------
    estimator : str
        Name of the model, used in the error message when a covariance
        estimator is not supported by the model.
    *cov_estimators : type
        The covariance estimator classes that the model supports.

    Notes
    -----
    Indexing the manager with a ``cov_type`` name, e.g.,
    ``manager["robust"]``, returns the covariance estimator class. It raises an
    ``UnknownOptionError``, which is both a ``ValueError`` and a ``KeyError``,
    if the name is unknown or the estimator is not supported by the model.
    """

    COVARIANCE_ESTIMATORS: ClassVar[dict[str, CovarianceEstimatorType]] = {
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
        # The names that the model supports, not all of the names that are known
        supported = [
            name
            for name, estimator in self.COVARIANCE_ESTIMATORS.items()
            if estimator in self._supported
        ]
        name = lower_option(item)
        if name not in self.COVARIANCE_ESTIMATORS:
            raise unknown_option_error("cov_type", item, supported)
        cov_est = self.COVARIANCE_ESTIMATORS[name]
        if cov_est not in self._supported:
            problem = (
                "Requested covariance estimator is not supported "
                "for the {}.".format(self._estimator)
            )
            raise unknown_option_error("cov_type", item, supported, problem=problem)
        return cov_est


class FamaMacBethCovariance(HomoskedasticCovariance):
    """
    HAC estimator for Fama-MacBeth estimator.

    Parameters
    ----------
    y : ndarray
        (entity x time) by 1 stacked array of the dependent variable.
    x : ndarray
        (entity x time) by variables stacked array of the exogenous variables.
    params : ndarray
        (variables by 1) array of estimated model parameters.
    all_params : DataFrame
        (nobs by variables) array of all estimated model parameters.
    debiased : bool, optional
        Flag indicating whether to debias the estimator. The default is False.
    bandwidth : float, optional
        Non-negative bandwidth to use. It must be below T, the number of time
        periods that are used. The Bartlett and Parzen kernels use its integer
        part. Set to 0 to disable autocorrelation robustness. If not provided, it
        is computed from the T estimates and is at most T - 1. The computation
        needs at least 4 time periods with the Bartlett kernel, 5 with the Parzen
        kernel and 6 with the quadratic spectral kernel. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.
    kernel : str, optional
        Name of the kernel. If not provided, "newey-west", which is the Bartlett
        kernel, is used. The names are not case sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    Raises
    ------
    ValueError
        If the bandwidth is not below the number of time periods that are used.
        It is also raised if the bandwidth is not provided and there are fewer
        time periods than the kernel needs to compute it, or if the bandwidth
        that is computed is not finite.
    UnknownOptionError
        If ``kernel`` is not one of the supported names, whether or not the
        covariance is a kernel covariance. It is both a ``ValueError`` and a
        ``KeyError``.

    Notes
    -----
    Covariance is a Kernel covariance of all estimated parameters.

    The :math:`T` cross-sectional estimates, one for each time period that is
    used, have autocovariances up to lag :math:`T-1`, so the bandwidth must be
    below :math:`T`. The bandwidth is checked, or computed if it is not
    provided, when the estimator is created, so that the ``fit`` method of the
    model raises the error.

    .. versionchanged:: 7.1

       A bandwidth that is not below the number of time periods raises a
       ``ValueError``. Earlier versions raised "Length of w is larger than the
       number of elements in z" with the Bartlett and Parzen kernels, when the
       covariance was first used. The quadratic spectral kernel, which has no
       truncation, returned a covariance without an error. With a large bandwidth
       the weights of all of the lags are close to one, and the deviations of the
       estimates from their average sum to zero, so that the covariance was close
       to zero.

    .. versionchanged:: 7.1

       The bandwidth that is computed from the estimates needs at least 4 time
       periods with the Bartlett kernel, 5 with the Parzen kernel and 6 with the
       quadratic spectral kernel, and a ``ValueError`` is raised for fewer, when
       the estimator is created. The deviations of the estimates from their
       average sum to zero. With fewer periods the rule divides by the sum of
       all of their autocovariances, which is zero, so that earlier versions
       raised "cannot convert float infinity to integer" (an ``OverflowError``) or
       "cannot convert float NaN to integer" when the covariance was first used,
       or used a bandwidth of :math:`T-1` because of the rounding error of the
       sum.

    .. versionchanged:: 7.1

       A ``kernel`` that is not one of the supported names raises an
       ``UnknownOptionError`` when the estimator is created, so that the ``fit``
       method of the model raises, also if the bandwidth is given. Earlier versions
       raised it when the covariance was first used, for example when ``std_errors``
       of the results was read, so that ``fit`` returned results that raised when
       the covariance was used.
    """

    def __init__(
        self,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        all_params: DataFrame,
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
        # Checked here, also if the bandwidth is selected from the data, so that fit
        # raises and not the first use of the covariance
        self._kernel_weights = lookup_option(KERNEL_LOOKUP, self._kernel, "kernel")
        if bandwidth is not None:
            # Checked here so that fit raises, and not the first use of the covariance
            deviations = np.asarray(all_params) - params.T
            nperiods = int(np.all(np.isfinite(deviations), 1).sum())
            if bandwidth >= nperiods:
                raise ValueError(
                    "The bandwidth of the Fama-MacBeth kernel covariance, "
                    f"{bandwidth:g}, must be below the number of time periods that "
                    f"are used, {nperiods}. The {nperiods} cross-sectional estimates "
                    f"have autocovariances up to lag {nperiods - 1}. Use a smaller "
                    "bandwidth."
                )
        else:
            # Selected here as well, so that fit raises if it cannot be selected, and
            # not the first use of the covariance
            _ = self.bandwidth

    @cached_property
    def bandwidth(self) -> float:
        """Bandwidth of the kernel."""
        if self._bandwidth is None:
            all_params = np.asarray(self._all_params)
            e = all_params - self._params.T
            e = e[np.all(np.isfinite(e), 1)]
            nperiods = e.shape[0]
            minimum = _min_nobs_automatic_bandwidth(self._kernel)
            if nperiods < minimum:
                raise ValueError(
                    "The automatic bandwidth of the Fama-MacBeth kernel covariance "
                    f"with the {self._kernel!r} kernel needs at least {minimum} time "
                    "periods, but the number of time periods that are used is "
                    f"{nperiods}. Pass the bandwidth to fit, which must be below "
                    f"{nperiods}."
                )
            # Estimates that do not vary have a standard deviation of 0, and the series
            # is NaN. kernel_optimal_bandwidth raises, so there is no need to warn
            with np.errstate(divide="ignore", invalid="ignore"):
                stde = np.sum(e / e.std(0)[None, :], 1)
            self._bandwidth = kernel_optimal_bandwidth(stde, self._kernel)
        assert self._bandwidth is not None
        return self._bandwidth

    @cached_property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Estimated covariance."""
        e = np.asarray(self._all_params) - self._params.T
        e = e[np.all(np.isfinite(e), 1)]
        nobs = e.shape[0]

        bw = self.bandwidth
        w = self._kernel_weights(bw, nobs - 1)
        cov = cov_kernel(e, w)
        return cov / (nobs - int(bool(self._debiased)))

    @property
    def all_params(self) -> DataFrame:
        """
        The set of parameters estimated for each of the time periods.

        Returns
        -------
        DataFrame
            The parameters with shape (nobs, nparam).
        """
        return self._all_params


def check_covariance_kwargs(
    cov_estimators: CovarianceManager,
    cov_type: str,
    cov_config: Mapping[str, Any],
    cluster_options: tuple[str, ...] = ("cluster_entity", "cluster_time"),
    fit_options: tuple[str, ...] = ("extra_df",),
) -> CovarianceEstimatorType:
    """
    Check that the keyword arguments are supported by a covariance estimator.

    Parameters
    ----------
    cov_estimators : CovarianceManager
        The manager of the covariance estimators that the model supports.
    cov_type : str
        Name of the covariance estimator.
    cov_config : dict
        The keyword arguments of ``fit`` for the covariance estimator.
    cluster_options : tuple of str, optional
        The options that make clusters, which the model supports if the estimator
        takes clusters.
    fit_options : tuple of str, optional
        The keyword arguments that ``fit`` accepts with every estimator. The
        default is ``("extra_df",)``.

    Returns
    -------
    type
        The covariance estimator class that `cov_type` selects.

    Raises
    ------
    UnknownOptionError
        If `cov_type` is unknown or is not supported.
    ValueError
        If `cov_config` contains a key that the estimator does not support.
    """
    estimator = cov_estimators[cov_type]
    # fit also accepts its own options, and the options that make clusters
    accepted = (*estimator.ALLOWED_KWARGS, *fit_options)
    if "clusters" in estimator.ALLOWED_KWARGS:
        accepted = (*accepted, *cluster_options)
    name = str(cov_type).lower()
    check_cov_config(cov_config, name, {name: accepted}, KERNEL_LOOKUP)
    return estimator


def setup_covariance_estimator(
    cov_estimators: CovarianceManager,
    cov_type: str,
    y: linearmodels.typing.data.Float64Array,
    x: linearmodels.typing.data.Float64Array,
    params: linearmodels.typing.data.Float64Array,
    entity_ids: linearmodels.typing.data.AnyIntArray,
    time_ids: linearmodels.typing.data.AnyIntArray,
    *,
    debiased: bool = False,
    extra_df: int = 0,
    **cov_config: Any,
) -> HomoskedasticCovariance:
    """
    Construct the covariance estimator for a ``cov_type``.

    Parameters
    ----------
    cov_estimators : CovarianceManager
        The manager of the covariance estimators that the model supports.
    cov_type : str
        Name of the covariance estimator, one of the names supported by
        `cov_estimators`.
    y : ndarray
        (entity x time) by 1 stacked array of the dependent variable.
    x : ndarray
        (entity x time) by variables stacked array of the exogenous variables.
    params : ndarray
        Variables by 1 array of estimated model parameters.
    entity_ids : ndarray
        (entity x time) by 1 stacked array of entity ids.
    time_ids : ndarray
        (entity x time) by 1 stacked array of time ids.
    debiased : bool, optional
        Flag indicating whether to debias the estimator. The default is False.
    extra_df : int, optional
        Additional degrees of freedom consumed by models beyond the number of
        columns in x. The default is 0.
    **cov_config
        Keyword arguments of the covariance estimator. The keys that are
        allowed depend on the estimator: ``clusters`` and ``group_debias`` for
        the clustered estimator, ``kernel`` and ``bandwidth`` for the
        Driscoll-Kraay and autocorrelation robust estimators, and none for the
        homoskedastic and heteroskedastic estimators.

    Returns
    -------
    HomoskedasticCovariance
        The covariance estimator, an instance of the class that
        `cov_type` selects, which is a subclass of
        ``HomoskedasticCovariance``.

    Raises
    ------
    UnknownOptionError
        If `cov_type` is unknown or is not supported. It is both a
        ``ValueError`` and a ``KeyError``.
    ValueError
        If `cov_config` contains a key that the estimator does not support.
    TypeError
        If a value of `cov_config` does not have the type that its key requires.
    """
    estimator = check_covariance_kwargs(cov_estimators, cov_type, cov_config)
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
