"""
Covariance and weight estimation for GMM IV estimators.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

from numpy import asarray, errstate, ndarray, unique
from numpy.linalg import inv

from linearmodels.iv.covariance import (
    KERNEL_LOOKUP,
    HomoskedasticCovariance,
    as_cluster_array,
    kernel_optimal_bandwidth,
)
from linearmodels.shared.covariance import cov_cluster, cov_kernel
from linearmodels.shared.exceptions import (
    DefaultBandwidthWarning,
    lookup_option,
    lower_option,
    unknown_option_error,
)
import linearmodels.typing.data


def _uses_default_bandwidth(config: Mapping[str, Any]) -> bool:
    """Whether neither bandwidth nor optimal_bw selects the kernel bandwidth."""
    return config.get("bandwidth") is None and config.get("optimal_bw") is None


_WEIGHT_EXAMPLES = {
    "IVGMM": 'IVGMM(dependent, exog, endog, instruments, weight_type="kernel", '
    "optimal_bw=False)",
    "IVGMMCUE": 'IVGMMCUE(dependent, exog, endog, instruments, weight_type="kernel", '
    "optimal_bw=False)",
    "IVSystemGMM": 'IVSystemGMM(equations, weight_type="kernel", optimal_bw=False)',
}


def _default_bandwidth_message(
    *, weights: bool, covariance: bool, model: str = "IVGMM"
) -> str:
    """Message of the warning that the default bandwidth changed in 7.1."""
    used, restore = [], []
    if weights:
        used.append('the weight matrix (weight_type="kernel")')
        restore.append(
            "in the weight configuration for the weight matrix, for example "
            + _WEIGHT_EXAMPLES[model]
        )
    if covariance:
        used.append('the parameter covariance (cov_type="kernel")')
        restore.append(
            "to fit for the covariance, for example "
            'fit(cov_type="kernel", optimal_bw=False)'
        )
    results = (
        "estimates, standard errors and J statistic" if weights else "standard errors"
    )
    return (
        "The default bandwidth of the kernel estimators of the covariance of the "
        "moment conditions changed in linearmodels 7.1. Neither bandwidth nor "
        f"optimal_bw was given, so the bandwidth of {' and '.join(used)} is "
        "estimated from the data. In 7.0 and earlier it was nobs - 2, which is not "
        f"a consistent estimator, so the {results} can differ from those of "
        "earlier versions. To restore the previous behavior, pass optimal_bw=False "
        "(or bandwidth=nobs - 2, where nobs is the number of observations used in "
        f"the estimation) {', and '.join(restore)}. Passing bandwidth or "
        "optimal_bw silences this warning, and optimal_bw=True selects the "
        "bandwidth from the data without a warning. This warning will be removed "
        "in linearmodels 8.0."
    )


def _warn_default_bandwidth(
    *, weights: bool, covariance: bool, stacklevel: int, model: str = "IVGMM"
) -> None:
    """
    Warn that a kernel estimator uses the bandwidth selected by default.

    The default bandwidth was nobs - 2 before linearmodels 7.1.
    """
    warnings.warn(
        _default_bandwidth_message(weights=weights, covariance=covariance, model=model),
        DefaultBandwidthWarning,
        stacklevel=stacklevel,
    )


class HomoskedasticWeightMatrix:
    r"""
    Homoskedastic (unadjusted) weight estimation.

    Parameters
    ----------
    center : bool, optional
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix. The default is False.
    debiased : bool, optional
        Flag indicating whether to use small-sample adjustments. The default is
        False.

    Notes
    -----
    The weight matrix estimator is

    .. math::

      s^{2} & =n^{-1}\sum_{i=1}^{n}(\epsilon_i-\bar{\epsilon})^2 \\
      W & =n^{-1}s^{2}\sum_{i=1}^{n}z_i'z_i

    where :math:`z_i` contains both the exogenous regressors and instruments.

    ``center`` has no effect on this estimator since it is always centered.
    """

    def __init__(self, center: bool = False, debiased: bool = False) -> None:
        self._center = center
        self._debiased = debiased
        self._bandwidth: int | None = 0

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute the covariance of the moment conditions.

        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar).
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr).
        eps : ndarray
            Model errors (nobs by 1).

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape
        mu = eps.mean(0)
        s2 = (eps - mu).T @ (eps - mu) / nobs
        w = s2 * z.T @ z / nobs
        w *= 1 if not self._debiased else nobs / (nobs - nvar)
        return w

    @property
    def config(self) -> dict[str, str | bool | ndarray | int | None]:
        """
        Weight estimator configuration.

        Returns
        -------
        dict
            Dictionary containing weight estimator configuration information.
        """
        return {"center": self._center, "debiased": self._debiased}


class HeteroskedasticWeightMatrix(HomoskedasticWeightMatrix):
    r"""
    Heteroskedasticity robust weight estimation.

    Parameters
    ----------
    center : bool, optional
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix. The default is False.
    debiased : bool, optional
        Flag indicating whether to use small-sample adjustments. The default is
        False.

    Notes
    -----
    The weight matrix estimator is

    .. math::

      g_i & =z_i\epsilon_i\\
      W & =n^{-1}\sum_{i=1}^{n}g'_ig_i

    where :math:`z_i` contains both the exogenous regressors and instruments.
    """

    def __init__(self, center: bool = False, debiased: bool = False) -> None:
        super().__init__(center, debiased)

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute the covariance of the moment conditions.

        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar).
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr).
        eps : ndarray
            Model errors (nobs by 1).

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        w = ze.T @ ze / nobs
        w *= 1 if not self._debiased else nobs / (nobs - nvar)
        return w


class KernelWeightMatrix(HomoskedasticWeightMatrix):
    r"""
    Heteroskedasticity, autocorrelation robust weight estimation.

    Parameters
    ----------
    kernel : str, optional
        Name of kernel weighting function to use. The default is "bartlett". The
        names are not case sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    bandwidth : int, optional
        Bandwidth to use when computing kernel weights. If not provided, the
        bandwidth is selected by ``optimal_bw``.
    center : bool, optional
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix. The default is False.
    debiased : bool, optional
        Flag indicating whether to use small-sample adjustments. The default
        is False.
    optimal_bw : bool, optional
        Flag indicating whether to estimate the bandwidth from the data when
        ``bandwidth`` is not provided. If False, nobs - 2 is used. If True,
        the bandwidth is estimated using
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`, which
        needs at least 4 observations with the Bartlett kernel, 5 with the
        Parzen kernel and 6 with the quadratic spectral kernel. The default
        also estimates the bandwidth, but it issues a
        ``DefaultBandwidthWarning`` since it changed in 7.1.

    See Also
    --------
    linearmodels.iv.covariance.kernel_weight_bartlett : Kernel weights from a
        Bartlett kernel.
    linearmodels.iv.covariance.kernel_weight_parzen : Kernel weights from a Parzen
        kernel.
    linearmodels.iv.covariance.kernel_weight_quadratic_spectral : Kernel weights
        from a quadratic-spectral kernel.

    Notes
    -----
    The weight matrix is

    .. math::

      g_i & =z_i \epsilon_i \\
      W & =n^{-1}(\Gamma_0+\sum_{j=1}^{n-1}k(j)(\Gamma_j+\Gamma_j')) \\
      \Gamma_j & =\sum_{i=j+1}^n g'_i g_{i-j}

    where :math:`k(j)` is the kernel weight for lag j and :math:`z_i`
    contains both the exogenous regressors and instruments.

    If neither ``bandwidth`` nor ``optimal_bw`` is given, the bandwidth is
    estimated from the data every time that the weight matrix is computed.
    In 7.0 and earlier the default was ``nobs - 2``, which is not a consistent
    estimator of the covariance of the moment conditions. Using the default
    issues a ``DefaultBandwidthWarning`` (a ``UserWarning``), once for each
    instance. The previous behavior is restored by ``optimal_bw=False`` or by
    ``bandwidth=nobs - 2``. Passing ``bandwidth`` or ``optimal_bw`` silences the
    warning, and ``optimal_bw=True`` selects the new default without it.
    :class:`~linearmodels.iv.model.IVGMMCUE` estimates the bandwidth once, from
    the residuals of the first step estimates, and uses it in every evaluation
    of its objective function.

    .. versionchanged:: 7.1

       ``weight_matrix`` raises a ``ValueError`` if the bandwidth is estimated
       from the data and there are fewer observations than the kernel needs
       (see ``optimal_bw``), or if the estimated bandwidth is not finite. Pass
       ``bandwidth`` or ``optimal_bw=False`` to use the weight matrix with
       fewer observations. Earlier versions raised "cannot convert float
       infinity to integer" (an ``OverflowError``) or "cannot convert float NaN
       to integer" for some data when ``optimal_bw`` was True, and used a
       bandwidth of nobs - 1, or another bandwidth, for other data.
    """

    def __init__(
        self,
        kernel: str = "bartlett",
        bandwidth: int | None = None,
        center: bool = False,
        debiased: bool = False,
        optimal_bw: bool | None = None,
    ) -> None:
        super().__init__(center, debiased)
        self._bandwidth = bandwidth
        self._orig_bandwidth = bandwidth
        kernel = lower_option(kernel)
        self._kernel = kernel
        self._kernels = KERNEL_LOOKUP
        if isinstance(kernel, str) and kernel not in KERNEL_LOOKUP:
            raise unknown_option_error("kernel", kernel, KERNEL_LOOKUP)
        # Selecting the bandwidth from the data is the default since 7.1, and it
        # warns since the default was nobs - 2
        self._default_bandwidth = bandwidth is None and optimal_bw is None
        self._optimal_bw = True if optimal_bw is None else optimal_bw
        self._warned = False
        # A bandwidth that was estimated from the data once, and is used instead
        # of estimating it again each time that the weight matrix is computed
        self._fixed_bandwidth: int | None = None

    @property
    def _estimates_bandwidth(self) -> bool:
        """Whether the bandwidth is estimated from the data."""
        return self._orig_bandwidth is None and self._optimal_bw

    def _scores(
        self,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """The moment conditions, centered if the weight matrix is centered."""
        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu
        return ze

    def _estimate_bandwidth(self, ze: linearmodels.typing.data.Float64Array) -> int:
        """The bandwidth that is estimated from the moment conditions."""
        # A moment that does not vary has a standard deviation of 0, and the series is
        # NaN. kernel_optimal_bandwidth raises, so there is no need to warn
        with errstate(divide="ignore", invalid="ignore"):
            g = ze / ze.std(0)[None, :]
        g = g.sum(1)
        return kernel_optimal_bandwidth(g, self._kernel)

    def _fix_bandwidth(
        self,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> int:
        """
        Estimate the bandwidth from eps, and use it for every weight matrix.

        The bandwidth that is estimated from the data depends on the residuals.
        Estimating it at each evaluation of an objective function that depends
        on the parameters makes the objective a step function of them. It is
        estimated once instead, and then used each time that the weight matrix
        is computed.
        """
        self._fixed_bandwidth = self._estimate_bandwidth(self._scores(z, eps))
        return self._fixed_bandwidth

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute the covariance of the moment conditions.

        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar).
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr).
        eps : ndarray
            Model errors (nobs by 1).

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape
        ze = self._scores(z, eps)

        if self._estimates_bandwidth:
            if self._default_bandwidth and not self._warned:
                # Once for each instance, and not for each call
                self._warned = True
                _warn_default_bandwidth(weights=True, covariance=False, stacklevel=3)
            if self._fixed_bandwidth is None:
                self._bandwidth = self._estimate_bandwidth(ze)
            else:
                self._bandwidth = self._fixed_bandwidth
        elif self._orig_bandwidth is None:
            self._bandwidth = nobs - 2
        bw = self._bandwidth
        assert bw is not None
        w = lookup_option(self._kernels, self._kernel, "kernel")(bw, nobs - 1)

        s = cov_kernel(ze, w)
        s *= 1 if not self._debiased else nobs / (nobs - nvar)

        return s

    @property
    def config(self) -> dict[str, str | bool | ndarray | int | None]:
        """
        Weight estimator configuration.

        Returns
        -------
        dict
            Dictionary containing weight estimator configuration information.
        """
        return {
            "center": self._center,
            "bandwidth": self._bandwidth,
            "kernel": self._kernel,
            "debiased": self._debiased,
        }

    @property
    def bandwidth(self) -> int | None:
        """Actual bandwidth used in estimating the weight matrix."""
        return self._bandwidth


class OneWayClusteredWeightMatrix(HomoskedasticWeightMatrix):
    """
    Clustered (one-way) weight estimation.

    Parameters
    ----------
    clusters : array_like
        Cluster membership. Either a 1-d array with an element for each
        observation or a 2-d array with a single column, since only one-way
        clustering is supported. A ``ValueError`` is raised if it has more
        columns.
    center : bool, optional
        Flag indicating whether to center the moment conditions by subtracting
        the mean before computing the weight matrix. The default is False.
    debiased : bool, optional
        Flag indicating whether to use small-sample adjustments. The default is
        False.

    Raises
    ------
    TypeError
        If ``clusters`` is not array_like, or has a complex dtype.
    ValueError
        If ``clusters`` has more than one column.
    """

    def __init__(
        self,
        clusters: linearmodels.typing.data.AnyArray,
        center: bool = False,
        debiased: bool = False,
    ) -> None:
        super().__init__(center, debiased)
        self._clusters = as_cluster_array(clusters, 1)

    def weight_matrix(
        self,
        x: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Compute the covariance of the moment conditions.

        Parameters
        ----------
        x : ndarray
            Model regressors (exog and endog), (nobs by nvar).
        z : ndarray
            Model instruments (exog and instruments), (nobs by ninstr).
        eps : ndarray
            Model errors (nobs by 1).

        Returns
        -------
        ndarray
            Covariance of GMM moment conditions.
        """
        nobs, nvar = x.shape

        ze = z * eps
        mu = ze.mean(axis=0) if self._center else 0
        ze -= mu

        clusters = self._clusters
        if clusters.shape[0] != nobs:
            raise ValueError(
                "clusters has the wrong nobs. Expected {}, "
                "got {}".format(nobs, clusters.shape[0])
            )
        clusters = asarray(clusters).copy().squeeze()

        s = cov_cluster(ze, clusters)

        if self._debiased:
            num_clusters = len(unique(clusters))
            scale = (nobs - 1) / (nobs - nvar) * num_clusters / (num_clusters - 1)
            s *= scale

        return s

    @property
    def config(self) -> dict[str, str | bool | ndarray | int | None]:
        """
        Weight estimator configuration.

        Returns
        -------
        dict
            Dictionary containing weight estimator configuration information.
        """
        return {
            "center": self._center,
            "clusters": self._clusters,
            "debiased": self._debiased,
        }


class IVGMMCovariance(HomoskedasticCovariance):
    """
    Covariance estimation for GMM models.

    Parameters
    ----------
    x : ndarray
        Model regressors (nobs by nvar).
    y : ndarray
        Series modeled (nobs by 1).
    z : ndarray
        Instruments used for endogenous regressors (nobs by ninstr).
    params : ndarray
        Estimated model parameters (nvar by 1).
    w : ndarray
        Weighting matrix used in GMM estimation (ninstr by ninstr).
    cov_type : str, optional
        Covariance estimator to use. The default is "robust". The names are not
        case sensitive. Supported covariance estimators are:

        * "unadjusted", "homoskedastic" - Assumes moment conditions are
          homoskedastic
        * "robust", "heteroskedastic" - Allows for heteroskedasticity but not
          autocorrelation
        * "kernel" - Allows for heteroskedasticity and autocorrelation
        * "clustered" - Allows for one-way cluster dependence

    debiased : bool, optional
        Flag indicating whether to debias the covariance estimator. The default
        is False.
    **cov_config : dict
        Optional keyword arguments that are specific to a particular cov_type.
        The keys are ``center`` (bool) for every covariance estimator,
        ``kernel`` (str), ``bandwidth`` (int) and ``optimal_bw`` (bool) for
        "kernel", and ``clusters`` (array_like) for "clustered". The names of the
        kernels are not case sensitive. See Notes.

    Raises
    ------
    ValueError
        If the number of rows of x, y and z differ, or if x and params have
        incompatible dimensions.
    UnknownOptionError
        If ``cov_type`` is not supported. It is both a ``KeyError`` and a
        ``ValueError``.

    See Also
    --------
    linearmodels.iv.covariance.HomoskedasticCovariance : Homoskedastic covariance.
    linearmodels.iv.covariance.HeteroskedasticCovariance : Heteroskedasticity
        robust covariance.
    linearmodels.iv.covariance.KernelCovariance : Kernel (HAC) covariance.
    linearmodels.iv.covariance.ClusteredCovariance : Clustered covariance.

    Notes
    -----
    Optional keyword argument for specific covariance estimators:

    **kernel**

    * ``kernel``: Name of kernel to use. The names are not case sensitive. See
      :class:`~linearmodels.iv.covariance.KernelCovariance` for details on
      available kernels
    * ``bandwidth``: Bandwidth to use when computing the weight.  If not
      provided, it is selected by ``optimal_bw``.
    * ``optimal_bw``: Flag indicating whether to estimate the bandwidth from
      the data when ``bandwidth`` is not provided. If False, nobs - 2 is used.
      If True, the bandwidth is estimated using
      :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`, which
      needs at least 4 observations with the Bartlett kernel, 5 with the Parzen
      kernel and 6 with the quadratic spectral kernel, and raises a
      ``ValueError`` for fewer.

    If neither ``bandwidth`` nor ``optimal_bw`` is given, the bandwidth is
    estimated from the data. In 7.0 and earlier the default was ``nobs - 2``,
    which is not a consistent estimator of the covariance of the moment
    conditions. Using the default issues a ``DefaultBandwidthWarning`` (a
    ``UserWarning``). The previous behavior is restored by ``optimal_bw=False``
    or by ``bandwidth=nobs - 2``. Passing ``bandwidth`` or ``optimal_bw``
    silences the warning, and ``optimal_bw=True`` selects the new default
    without it.

    **cluster**

    * ``clusters``: Array containing the cluster indices.  See
      :class:`~linearmodels.iv.covariance.ClusteredCovariance`
    """

    # TODO: 2-way clustering
    def __init__(
        self,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        w: linearmodels.typing.data.Float64Array,
        cov_type: str = "robust",
        debiased: bool = False,
        **cov_config: str | bool,
    ) -> None:
        super().__init__(x, y, z, params, debiased)
        cov_type = lower_option(cov_type)
        self._cov_type = cov_type
        self._cov_config = cov_config
        self.w = w
        self._bandwidth = cov_config.get("bandwidth", None)
        self._kernel = cov_config.get("kernel", "")
        self._name = "GMM Covariance"
        if cov_type in ("robust", "heteroskedastic"):
            score_cov_estimator: Any = HeteroskedasticWeightMatrix
        elif cov_type in ("unadjusted", "homoskedastic"):
            score_cov_estimator = HomoskedasticWeightMatrix
        elif cov_type == "clustered":
            score_cov_estimator = OneWayClusteredWeightMatrix
        elif cov_type == "kernel":
            score_cov_estimator = KernelWeightMatrix
        else:
            supported = (
                "clustered",
                "heteroskedastic",
                "homoskedastic",
                "kernel",
                "robust",
                "unadjusted",
            )
            raise unknown_option_error("cov_type", cov_type, supported)
        self._score_cov_estimator = score_cov_estimator

    def __str__(self) -> str:
        out = super().__str__()
        cov_type = self._cov_type
        if cov_type in ("robust", "heteroskedastic"):
            out += "\nRobust (Heteroskedastic)"
        elif cov_type in ("unadjusted", "homoskedastic"):
            out += "\nUnadjusted (Homoskedastic)"
        elif cov_type == "clustered":
            out += "\nClustered (One-way)"
            clusters = self._cov_config.get("clusters", None)
            if clusters is not None:
                nclusters = len(unique(asarray(clusters)))
                out += f"\nNum Clusters: {nclusters}"
        else:  # kernel
            out += "\nKernel (HAC)"
            if self._cov_config.get("kernel", False):
                out += "\nKernel: {}".format(self._cov_config["kernel"])
            if self._cov_config.get("bandwidth", False):
                out += "\nBandwidth: {}".format(self._cov_config["bandwidth"])
        return out

    @property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Return the covariance of the estimated parameters."""
        x, z, eps, w = self.x, self.z, self.eps, self.w
        nobs = x.shape[0]
        xpz = x.T @ z / nobs
        xpzw = xpz @ w
        xpzwzpx_inv = inv(xpzw @ xpz.T)

        cov_config = self._cov_config
        if self._cov_type == "kernel" and _uses_default_bandwidth(cov_config):
            # The warning is for the covariance, and not for the weight estimator
            _warn_default_bandwidth(weights=False, covariance=True, stacklevel=3)
            cov_config = {**cov_config, "optimal_bw": True}
        score_cov = self._score_cov_estimator(debiased=self.debiased, **cov_config)
        s = score_cov.weight_matrix(x, z, eps)
        self._cov_config = score_cov.config

        c = xpzwzpx_inv @ (xpzw @ s @ xpzw.T) @ xpzwzpx_inv / nobs
        return (c + c.T) / 2

    @property
    def config(self) -> dict[str, str | bool | ndarray | int | None]:
        """Return the options of the covariance estimator as a dict."""
        conf: dict[str, str | bool | ndarray | int | None] = {"debiased": self.debiased}
        conf.update(self._cov_config)
        return conf
