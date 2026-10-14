"""
Covariance estimation for 2SLS and LIML IV estimators.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union, cast

from mypy_extensions import VarArg
from numpy import (
    arange,
    asarray,
    ceil,
    cos,
    empty,
    errstate,
    int64,
    isfinite,
    ndarray,
    ones,
    pi,
    sin,
    sum as npsum,
    unique,
    zeros,
)
from numpy.linalg import inv, pinv

from linearmodels.shared.covariance import cov_cluster, cov_kernel
from linearmodels.shared.exceptions import lower_option, unknown_option_error
from linearmodels.shared.inputs import as_ndarray
from linearmodels.shared.linalg import has_constant
import linearmodels.typing
import linearmodels.typing.data

KernelWeight = Union[
    Callable[[float, float], ndarray],
    Callable[[float, int], ndarray],
    Callable[[float, VarArg(Any)], ndarray],
]

CLUSTER_ERR = """\
clusters has the wrong nobs. Expected {0}, got {1}. Observations with a missing
value in the regression variables are dropped when the model is created, so
clusters must have one row for each of the remaining observations. When using a
clustered covariance estimator, drop the observations with missing values from
the data and from clusters before estimating the model."""

CLUSTER_COLUMNS_ERR = """\
clusters must be a 1-d array or a 2-d array with {0}, but its shape is {1}."""


def check_cluster_columns(
    clusters: linearmodels.typing.data.AnyArray, max_columns: int
) -> None:
    """
    Check that clusters has the number of variables that an estimator supports.

    Parameters
    ----------
    clusters : ndarray
        Cluster membership, either a 1-d array or an array with a column for
        each of the variables to cluster on.
    max_columns : int
        The number of clustering variables that the estimator supports, 1 or 2.

    Raises
    ------
    ValueError
        If clusters has more than 2 dimensions or does not have 1 to
        max_columns columns.
    """
    if clusters.ndim == 1 or (
        clusters.ndim == 2 and 1 <= clusters.shape[1] <= max_columns
    ):
        return
    supported = (
        "a single column (one-way clustering)"
        if max_columns == 1
        else "1 or 2 columns (one-way or two-way clustering)"
    )
    raise ValueError(CLUSTER_COLUMNS_ERR.format(supported, clusters.shape))


def as_cluster_array(
    clusters: linearmodels.typing.data.ArrayLike, max_columns: int
) -> linearmodels.typing.data.AnyArray:
    """
    Convert clusters to an array and check the number of variables.

    Parameters
    ----------
    clusters : array_like
        Cluster membership, either a 1-d array or an array with a column for
        each of the variables to cluster on.
    max_columns : int
        The number of clustering variables that the estimator supports, 1 or 2.

    Returns
    -------
    ndarray
        The cluster membership as an array.

    Raises
    ------
    TypeError
        If clusters is not array_like, or has a complex dtype.
    ValueError
        If clusters has more than 2 dimensions or does not have 1 to
        max_columns columns.
    """
    clusters = as_ndarray(clusters, "clusters")
    check_cluster_columns(clusters, max_columns)
    return clusters


def kernel_weight_bartlett(
    bw: float, *args: int
) -> linearmodels.typing.data.Float64Array:
    r"""
    Kernel weights from a Bartlett kernel.

    Parameters
    ----------
    bw : int
        Maximum lag to use in the kernel.
    *args : int
        Ignored. Accepted so that every kernel weight function can be called
        with the same arguments.

    Returns
    -------
    ndarray
        Weight array ordered by lag position (maxlag + 1).

    Notes
    -----
    .. math::

       w_i = 1 - i / (m + 1), \, i < m
    """
    return 1 - arange(int(bw) + 1) / (int(bw) + 1)


def kernel_weight_quadratic_spectral(
    bw: float, n: int
) -> linearmodels.typing.data.Float64Array:
    r"""
    Kernel weights from a quadratic-spectral kernel.

    Parameters
    ----------
    bw : int or float
        Bandwidth of the kernel.
    n : int
        Positive number of weights to return.

    Returns
    -------
    ndarray
        Weight array ordered by lag position (n + 1).

    Notes
    -----
    Unlike the Bartlett or Parzen kernels, the QS kernel is not truncated at
    a specific lag, and so weights are computed for all available lags in
    the sample.

    .. math::

       z_i & = 6 \pi (i / m) / 5                                \\
       w_0 &  = 1                                                \\
       w_i &  = 3(\sin(z_i)/z_i - cos(z_i))/z_i^ 2, \, i \geq 1

    where m is the bandwidth.
    """
    if bw == 0:  # Special handling of bw == 0 case
        w = zeros(n + 1)
        w[0] = 0
        return w

    z = arange(n + 1).astype(float) / float(bw)
    w = cast("linearmodels.typing.data.FloatArray1D", 6 * pi * z / 5)
    w[0] = 1
    w[1:] = 3 / w[1:] ** 2 * (sin(w[1:]) / w[1:] - cos(w[1:]))

    return w


def kernel_weight_parzen(
    bw: float, *args: int
) -> linearmodels.typing.data.Float64Array:
    r"""
    Kernel weights from a Parzen kernel.

    Parameters
    ----------
    bw : int
        Maximum lag to use in the kernel.
    *args : int
        Ignored. Accepted so that every kernel weight function can be called
        with the same arguments.

    Returns
    -------
    ndarray
        Weight array ordered by lag position (maxlag + 1).

    Notes
    -----
    .. math::

       z_i & = i / (m+1)                    \\
       w_i &  = 1-6z_i^2+6z_i^3, z \leq 0.5 \\
       w_i &  = 2(1-z_i)^3, z > 0.5
    """
    z = arange(int(bw) + 1) / (int(bw) + 1)
    w = 1 - 6 * z**2 + 6 * z**3
    w[z > 0.5] = 2 * (1 - z[z > 0.5]) ** 3
    return w


def _bandwidth_rule(kernel: str) -> tuple[int, float, float]:
    """
    The constants of the automatic bandwidth rule of a kernel.

    Parameters
    ----------
    kernel : str
        Name of the kernel.

    Returns
    -------
    q : int
        The characteristic exponent of the kernel.
    c : float
        The constant of the rule.
    exponent : float
        The exponent of t / 100 in the number of lags of the pilot estimates.

    Raises
    ------
    UnknownOptionError
        If kernel is not one of the supported names.
    """
    kernel = lower_option(kernel)
    if kernel in ("bartlett", "newey-west"):
        return 1, 1.1447, 2 / 9
    if kernel in ("qs", "andrews", "quadratic-spectral"):
        return 2, 1.3221, 2 / 25
    if kernel in ("gallant", "parzen"):
        return 2, 2.6614, 4 / 25
    raise unknown_option_error("kernel", kernel, KERNEL_LOOKUP)


def _pilot_lags(nobs: int, exponent: float) -> int:
    """The number of lags in the pilot estimates of the automatic bandwidth."""
    return int(ceil(4 * (nobs / 100) ** exponent))


def _min_nobs_automatic_bandwidth(kernel: str) -> int:
    """
    The fewest observations for which the automatic bandwidth of a kernel exists.

    The pilot estimates use the autocovariances up to the pilot lag, and the pilot
    lag has to be below nobs - 1. It grows by at most 1 for each additional
    observation, so that every larger number of observations works too.

    Parameters
    ----------
    kernel : str
        Name of the kernel.

    Returns
    -------
    int
        The smallest number of observations with a pilot lag below nobs - 1.

    Raises
    ------
    UnknownOptionError
        If kernel is not one of the supported names.
    """
    exponent = _bandwidth_rule(kernel)[2]
    nobs = 1
    while _pilot_lags(nobs, exponent) >= nobs - 1:
        nobs += 1
    return nobs


def kernel_optimal_bandwidth(
    x: linearmodels.typing.data.Float64Array, kernel: str = "bartlett"
) -> int:
    r"""
    Select the bandwidth of a kernel covariance estimator from the data.

    Parameters
    ----------
    x : ndarray
        The series to use when computing the bandwidth. It has t observations,
        and is a 1-d array of length t or an array with t rows and 1 column.
    kernel : str, optional
        Name of the kernel to use. The default is "bartlett". The names are not
        case sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    Returns
    -------
    int
        Optimal bandwidth. Set to nobs - 1 if computed bandwidth is larger.

    Raises
    ------
    ValueError
        If x has fewer observations than the kernel needs: 4 for the Bartlett
        kernel, 5 for the Parzen kernel and 6 for the quadratic spectral kernel.
        It is also raised if the bandwidth is not finite, which happens if x has
        values that are not finite or if the estimate of the long-run variance
        that the bandwidth divides by is exactly 0, as it is for a series of zeros.
    UnknownOptionError
        If kernel is not one of the supported names. It is both a ``KeyError``
        and a ``ValueError``.

    See Also
    --------
    kernel_weight_bartlett : Kernel weights from a Bartlett kernel.
    kernel_weight_parzen : Kernel weights from a Parzen kernel.
    kernel_weight_quadratic_spectral : Kernel weights from a quadratic-spectral
        kernel.

    Notes
    -----
    The bandwidth is that of the nonparametric rule of Newey and West [1]_. With
    the autocovariances of the series :math:`x_i`,

    .. math::

       \hat{\sigma}_j = t^{-1} \sum_{i=j+1}^{t} x_i x_{i-j},

    the rule computes two weighted sums of the first :math:`n` of them,

    .. math::

       \hat{s}_0 & = \hat{\sigma}_0 + 2 \sum_{j=1}^{n} \hat{\sigma}_j \\
       \hat{s}_q & = 2 \sum_{j=1}^{n} j^q \hat{\sigma}_j

    and the bandwidth is the smaller of :math:`\lceil \hat{m} \rceil` and
    :math:`t - 1`, where

    .. math::

       \hat{m} = c \left\{ \left(\hat{s}_q / \hat{s}_0 \right)^2 \right\}^{1 / (2q + 1)}
                 t^{1 / (2q + 1)}.

    The exponent :math:`q` and the constant :math:`c` depend on the kernel. The
    number of lags in the pilot estimates, :math:`\hat{s}_0` and :math:`\hat{s}_q`,
    is :math:`n = \lceil 4 (t / 100)^a \rceil`, and the exponent :math:`a`
    depends on the kernel too:

    ===================================== = ====== ===== ============
    Kernel                                q c      a     Fewest obs.
    ===================================== = ====== ===== ============
    "bartlett", "newey-west"              1 1.1447 2/9   4
    "parzen", "gallant"                   2 2.6614 4/25  5
    "qs", "quadratic-spectral", "andrews" 2 1.3221 2/25  6
    ===================================== = ====== ===== ============

    The autocovariances of a series with :math:`t` observations are defined up
    to lag :math:`t - 1`. If :math:`n \geq t - 1`, :math:`\hat{s}_0` is the sum
    of all of them, which is :math:`(\sum_i x_i)^2 / t`. It is 0, up to rounding
    error, for a series with a mean of 0, such as the scores of a least squares
    regression or the deviations of estimates from their average, so that the
    bandwidth divides by 0 or by rounding error. For any other series it depends
    on the mean alone, and not on the autocorrelation. The rule needs
    :math:`n < t - 1`. That is the case with at least 4 observations for the
    Bartlett kernel (:math:`n = 2`), 5 for the Parzen kernel (:math:`n = 3`) and
    6 for the quadratic spectral kernel (:math:`n = 4`), and a ``ValueError`` is
    raised for fewer. Pass the bandwidth to the estimator that uses the kernel to
    use it with fewer observations.

    .. versionchanged:: 7.1

       A ``ValueError`` is raised if the series has fewer observations than the
       kernel needs, and if the bandwidth is not finite. Earlier versions raised
       "cannot convert float infinity to integer" (an ``OverflowError``) or
       "cannot convert float NaN to integer" for some series, and returned
       :math:`t - 1`, or another number, for the others.

    References
    ----------
    .. [1] Newey, W. K. and West, K. D. (1994). Automatic lag selection in
       covariance matrix estimation. The Review of Economic Studies, 61(4),
       631-653.
    """
    t = x.shape[0]
    x = x.squeeze()
    q, c, exponent = _bandwidth_rule(kernel)
    m_star = _pilot_lags(t, exponent)
    if m_star >= t - 1:
        raise ValueError(
            f"The automatic bandwidth of the {kernel!r} kernel needs at least "
            f"{_min_nobs_automatic_bandwidth(kernel)} observations, but there are "
            f"{t}. Pass the bandwidth explicitly."
        )
    rate = 1 / (2 * q + 1)
    # Values that are not finite, or a long-run variance of 0, are reported by the
    # error below and not by warnings
    with errstate(divide="ignore", invalid="ignore", over="ignore"):
        sigma = empty(m_star + 1)
        sigma[0] = x.T @ x / t
        for i in range(1, m_star + 1):
            sigma[i] = x[i:].T @ x[:-i] / t
        s0 = sigma[0] + 2 * sigma[1:].sum()
        sq = 2 * npsum(sigma[1:] * arange(1, m_star + 1) ** q)
        gamma = c * ((sq / s0) ** 2) ** rate
        m = gamma * t**rate
    if not isfinite(m):
        raise ValueError(
            f"The automatic bandwidth of the {kernel!r} kernel is not finite. It is "
            "not defined if the data have values that are not finite, or if the "
            "estimate of their long-run variance, which the bandwidth divides by, "
            "is 0, for example because the data are all 0. Pass the bandwidth "
            "explicitly."
        )
    return min(int(ceil(m)), t - 1)


KERNEL_LOOKUP: dict[str, KernelWeight] = {
    "bartlett": kernel_weight_bartlett,
    "newey-west": kernel_weight_bartlett,
    "quadratic-spectral": kernel_weight_quadratic_spectral,
    "qs": kernel_weight_quadratic_spectral,
    "andrews": kernel_weight_quadratic_spectral,
    "gallant": kernel_weight_parzen,
    "parzen": kernel_weight_parzen,
}


class HomoskedasticCovariance:
    r"""
    Covariance estimation for homoskedastic data.

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
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment. The default is
        False.
    kappa : float, optional
        Value of kappa in the k-class estimator. The default is 1.
    extra_df : int, optional
        Degrees of freedom that the model uses in addition to the nvar columns
        of x, for example the absorbed effects of AbsorbingLS. The default is 0.
        It is only used if ``debiased`` is True, when n / (n-k) becomes
        n / (n-k-extra_df).

    Raises
    ------
    ValueError
        If the number of rows of x, y and z differ, if x and params have
        incompatible dimensions or if ``debiased`` is True and there are no more
        observations than the columns of x plus ``extra_df``.

    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} s^2 V^{-1}

    where

    .. math::

      s^2 = n^{-1} \sum_{i=1}^n \hat{\epsilon}_i^2

    If ``debiased`` is true, then :math:`s^2` is scaled by n / (n-k-df), where k is
    the number of columns of x and df is ``extra_df``.

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.
    """

    def __init__(
        self,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        debiased: bool = False,
        kappa: linearmodels.typing.Numeric = 1,
        extra_df: int = 0,
    ):
        if not (x.shape[0] == y.shape[0] == z.shape[0]):
            raise ValueError("x, y and z must have the same number of rows")
        if not x.shape[1] == len(params):
            raise ValueError("x and params must have compatible dimensions")

        self.x = x
        self.y = y
        self.z = z
        self.params = params
        self._debiased = debiased
        self.eps = y - x @ params
        self._kappa = kappa
        self._pinvz = pinv(z)
        nobs, nvar = x.shape
        if self._debiased and extra_df > 0 and nobs - nvar - extra_df <= 0:
            raise ValueError(
                "The small-sample adjustment of a debiased covariance needs more "
                f"observations ({nobs}) than the {nvar} columns of x and the "
                f"{extra_df} degrees of freedom that are used in addition."
            )
        self._scale: float = nobs / (nobs - nvar - extra_df) if self._debiased else 1.0
        self._name = "Unadjusted Covariance (Homoskedastic)"

    def __str__(self) -> str:
        out = self._name
        out += f"\nDebiased: {self._debiased}"
        if self._kappa != 1:
            out += f"\nKappa: {self._kappa:0.3f}"
        return out

    def __repr__(self) -> str:
        return (
            self.__str__() + "\n" + self.__class__.__name__ + f", id: {hex(id(self))}"
        )

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """Return the score covariance estimate."""
        x, z, eps = self.x, self.z, self.eps
        nobs = x.shape[0]
        s2 = eps.T @ eps / nobs
        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        if self._kappa != 1:
            kappa = self._kappa
            xpx = x.T @ x / nobs
            v = (1 - kappa) * xpx + kappa * v

        return self._scale * s2 * v

    @property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Return the covariance of the estimated parameters."""
        x, z = self.x, self.z
        nobs = x.shape[0]

        pinvz = self._pinvz
        v = (x.T @ z) @ (pinvz @ x) / nobs
        if self._kappa != 1:
            kappa = self._kappa
            xpx = x.T @ x / nobs
            v = (1 - kappa) * xpx + kappa * v

        vinv = inv(v)
        c = vinv @ self.s @ vinv / nobs
        return (c + c.T) / 2

    @property
    def s2(self) -> linearmodels.typing.data.Float64Array:
        """Return the variance of the residuals, adjusted if debiased."""
        nobs = self.x.shape[0]
        eps = self.eps

        return self._scale * eps.T @ eps / nobs

    @property
    def debiased(self) -> bool:
        """Return a flag indicating if the covariance is debiased."""
        return self._debiased

    @property
    def config(self) -> dict[str, Any]:
        """Return the options of the covariance estimator as a dict."""
        return {"debiased": self.debiased, "kappa": self._kappa}


class HeteroskedasticCovariance(HomoskedasticCovariance):
    r"""
    Covariance estimation for heteroskedastic data.

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
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment. The default is
        False.
    kappa : float, optional
        Value of kappa in the k-class estimator. The default is 1.
    extra_df : int, optional
        Degrees of freedom that the model uses in addition to the nvar columns
        of x, for example the absorbed effects of AbsorbingLS. The default is 0.
        It is only used if ``debiased`` is True, when n / (n-k) becomes
        n / (n-k-extra_df).

    Raises
    ------
    ValueError
        If the number of rows of x, y and z differ, if x and params have
        incompatible dimensions or if ``debiased`` is True and there are no more
        observations than the columns of x plus ``extra_df``.

    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} V^{-1} \hat{S} V^{-1}

    where

    .. math::

      \hat{S} = n^{-1} \sum_{i=1}^n \hat{\epsilon}_i^2 \hat{x}_i^{\prime} \hat{x}_i

    where :math:`\hat{\gamma}=(Z'Z)^{-1}(Z'X)` and
    :math:`\hat{x}_i = z_i\hat{\gamma}`. If ``debiased`` is true, then
    :math:`S` is scaled by n / (n-k-df), where k is the number of columns of x and
    df is ``extra_df``.

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.
    """

    def __init__(
        self,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        debiased: bool = False,
        kappa: linearmodels.typing.Numeric = 1,
        extra_df: int = 0,
    ):
        super().__init__(x, y, z, params, debiased, kappa, extra_df)
        self._name = "Robust Covariance (Heteroskedastic)"

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """Return the heteroskedasticity-robust score covariance estimate."""
        x, z, eps = self.x, self.z, self.eps
        nobs = x.shape[0]
        pinvz = self._pinvz
        xhat_e = z @ (pinvz @ x) * eps
        s = xhat_e.T @ xhat_e / nobs

        return self._scale * s


class KernelCovariance(HomoskedasticCovariance):
    r"""
    Kernel weighted (HAC) covariance estimation.

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
    kernel : str, optional
        Kernel name. The default is "bartlett". The names are not case
        sensitive. Supported kernels are:

        * "bartlett", "newey-west" - Bartlett's kernel
        * "parzen", "gallant" - Parzen's kernel
        * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

    bandwidth : int, optional
        Non-negative bandwidth to use with kernel. If not provided, automatic
        bandwidth selection is used. It needs at least 4 observations with the
        Bartlett kernel, 5 with the Parzen kernel and 6 with the quadratic
        spectral kernel, and a ``ValueError`` is raised for fewer when the
        covariance is first used. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment. The default is
        False.
    kappa : float, optional
        Value of kappa in the k-class estimator. The default is 1.
    extra_df : int, optional
        Degrees of freedom that the model uses in addition to the nvar columns
        of x, for example the absorbed effects of AbsorbingLS. The default is 0.
        It is only used if ``debiased`` is True, when n / (n-k) becomes
        n / (n-k-extra_df).

    Raises
    ------
    ValueError
        If the number of rows of x, y and z differ, if x and params have
        incompatible dimensions or if ``debiased`` is True and there are no more
        observations than the columns of x plus ``extra_df``.
    UnknownOptionError
        If ``kernel`` is not one of the supported kernels. It is both a
        ``KeyError`` and a ``ValueError``.

    See Also
    --------
    kernel_weight_bartlett : Kernel weights from a Bartlett kernel.
    kernel_weight_parzen : Kernel weights from a Parzen kernel.
    kernel_weight_quadratic_spectral : Kernel weights from a quadratic-spectral
        kernel.

    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} V^{-1} \hat{S} V^{-1}

    where

    .. math::

      \hat{S}_0 & = n^{-1} \sum_{i=1}^{n} \hat{\epsilon}^2_i \hat{x}_i^{\prime}
           \hat{x}_{i} \\
      \hat{S}_j & = n^{-1} \sum_{i=1}^{n-j}
          \hat{\epsilon}_i\hat{\epsilon}_{i+j} (\hat{x}_i^{\prime}
          \hat{x}_{i+j} + \hat{x}_{i+j}^{\prime} \hat{x}_{i}) \\
      \hat{S}   & = \sum_{i=0}^{bw} K(i, bw) \hat{S}_i

    where :math:`\hat{\gamma}=(Z'Z)^{-1}(Z'X)`,
    :math:`\hat{x}_i = z_i\hat{\gamma}` and :math:`K(i,bw)` is a weight that
    depends on the kernel. If ``debiased`` is true, then :math:`S` is scaled
    by n / (n-k-df), where k is the number of columns of x and df is ``extra_df``.

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.

    The automatic bandwidth is that of
    :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth` for the sum of
    the columns of :math:`\hat{\epsilon}_i \hat{x}_i`, without the column of the
    constant if :math:`\hat{x}` has one and other columns.

    .. versionchanged:: 7.1

       The automatic bandwidth raises a ``ValueError`` if there are fewer
       observations than the kernel needs, and if it is not finite. Earlier
       versions raised "cannot convert float infinity to integer" (an
       ``OverflowError``) or "cannot convert float NaN to integer" for some
       series, and used a bandwidth of :math:`n - 1`, or another bandwidth, for
       the others.
    """

    def __init__(
        self,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        kernel: str = "bartlett",
        bandwidth: linearmodels.typing.OptionalNumeric = None,
        debiased: bool = False,
        kappa: linearmodels.typing.Numeric = 1,
        extra_df: int = 0,
    ):
        super().__init__(x, y, z, params, debiased, kappa, extra_df)
        self._kernels = KERNEL_LOOKUP
        kernel = lower_option(kernel)
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._auto_bandwidth = False
        self._name = "Kernel Covariance (HAC)"

        if kernel not in KERNEL_LOOKUP:
            raise unknown_option_error("kernel", kernel, KERNEL_LOOKUP)

    def __str__(self) -> str:
        out = super().__str__()
        out += f"\nKernel: {self._kernel}"
        out += f"\nAutomatic Bandwidth: {self._auto_bandwidth}"
        if self._bandwidth:
            out += f"\nBandwidth: {self._bandwidth}"
        return out

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """Return the HAC score covariance estimate."""
        x, z, eps = self.x, self.z, self.eps
        nobs = x.shape[0]

        pinvz = self._pinvz
        xhat = z @ (pinvz @ x)
        xhat_e = asarray(xhat * eps, dtype=float)

        kernel = self.config["kernel"]
        bw = self.config["bandwidth"]
        if bw is None:
            self._auto_bandwidth = True

            const, loc = has_constant(xhat)
            sel = ones((xhat.shape[1], 1))
            if const and xhat.shape[1] > 1:
                sel[loc] = 0
            scores = xhat_e @ sel
            bw = kernel_optimal_bandwidth(scores, kernel)

        self._bandwidth = bw
        w = self._kernels[kernel](bw, nobs - 1)
        s = cov_kernel(xhat_e, w)

        return cast("ndarray", self._scale * s)

    @property
    def config(self) -> dict[str, Any]:
        """Return the options of the covariance estimator as a dict."""
        return {
            "debiased": self.debiased,
            "bandwidth": self._bandwidth,
            "kernel": self._kernel,
            "kappa": self._kappa,
        }


class ClusteredCovariance(HomoskedasticCovariance):
    r"""
    Covariance estimation for clustered data.

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
    clusters : array_like, optional
        Cluster group assignment, with shape nobs by ncluster where ncluster is
        1 or 2. If not provided, every observation is its own cluster. A
        ``ValueError`` is raised for more columns.
    debiased : bool, optional
        Flag indicating whether to use a small-sample adjustment. The default is
        False.
    kappa : float, optional
        Value of kappa in the k-class estimator. The default is 1.
    extra_df : int, optional
        Degrees of freedom that the model uses in addition to the nvar columns
        of x, for example the absorbed effects of AbsorbingLS. The default is 0.
        It is only used if ``debiased`` is True, when n / (n-k) becomes
        n / (n-k-extra_df).

    Raises
    ------
    ValueError
        If the number of rows of x, y and z differ, if x and params have
        incompatible dimensions, if clusters does not have one row for each
        observation or has more than 2 columns, or if ``debiased`` is True and
        there are no more observations than the columns of x plus ``extra_df``.

    Notes
    -----
    Covariance is estimated using

    .. math::

        n^{-1} V^{-1} \hat{S} V^{-1}

    where

    .. math::

      \hat{S} & = n^{-1} (G/(G-1)) \sum_{g=1}^G \xi_{g}^\prime \xi_{g} \\
      \xi_{g} & = \sum_{i\in\mathcal{G}_g} \hat{\epsilon}_i \hat{x}_i \\

    where :math:`\hat{\gamma}=(Z'Z)^{-1}(Z'X)` and
    :math:`\hat{x}_i = z_i\hat{\gamma}`.  :math:`\mathcal{G}_g` contains the
    indices of elements in cluster g. If ``debiased`` is true, then
    :math:`S` is scaled by g(n - 1) / ((g-1)(n-k-df)) where g is the number
    of groups, k is the number of columns of x and df is ``extra_df``.

    .. math::

      V = n^{-1} X'Z(Z'Z)^{-1}Z'X

    where :math:`X` is the matrix of variables included in the model and
    :math:`Z` is the matrix of instruments, including exogenous regressors.
    """

    def __init__(
        self,
        x: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        z: linearmodels.typing.data.Float64Array,
        params: linearmodels.typing.data.Float64Array,
        clusters: linearmodels.typing.data.AnyArray | None = None,
        debiased: bool = False,
        kappa: linearmodels.typing.Numeric = 1,
        extra_df: int = 0,
    ):
        super().__init__(x, y, z, params, debiased, kappa, extra_df)

        nobs = x.shape[0]
        clusters = arange(nobs) if clusters is None else clusters
        clusters = cast(
            "linearmodels.typing.data.AnyArray",
            as_ndarray(clusters, "clusters").squeeze(),
        )
        if clusters.shape[0] != nobs:
            raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))
        check_cluster_columns(clusters, 2)
        self._clusters = clusters
        if clusters.ndim == 1:
            self._num_clusters = [len(unique(clusters))]
            self._num_clusters_str = str(self._num_clusters[0])
        else:
            self._num_clusters = [
                len(unique(clusters[:, 0])),
                len(unique(clusters[:, 1])),
            ]
            self._num_clusters_str = ", ".join(map(str, self._num_clusters))
        if clusters is not None and clusters.shape[0] != nobs:
            raise ValueError(CLUSTER_ERR.format(nobs, clusters.shape[0]))
        self._name = "Clustered Covariance (One-Way)"

    def __str__(self) -> str:
        out = super().__str__()
        out += f"\nNum Clusters: {self._num_clusters_str}"
        return out

    @property
    def s(self) -> linearmodels.typing.data.Float64Array:
        """Return the clustered estimator of the score covariance."""

        def rescale(
            s: linearmodels.typing.data.Float64Array, nc: int, nobs: int
        ) -> linearmodels.typing.data.Float64Array:
            scale = float(self._scale * (nc / (nc - 1)) * ((nobs - 1) / nobs))
            return cast("ndarray", s * scale) if self.debiased else s

        x, z, eps = self.x, self.z, self.eps
        pinvz = self._pinvz
        xhat_e = asarray(z @ (pinvz @ x) * eps, dtype=float)

        nobs = x.shape[0]
        clusters = self._clusters
        if self._clusters.ndim == 1:
            s = cov_cluster(xhat_e, clusters)
            s = rescale(s, self._num_clusters[0], nobs)
        else:
            s0 = cov_cluster(xhat_e, clusters[:, 0].squeeze())
            s0 = rescale(s0, self._num_clusters[0], nobs)

            s1 = cov_cluster(xhat_e, clusters[:, 1].squeeze())
            s1 = rescale(s1, self._num_clusters[1], nobs)

            c0 = clusters[:, 0] - clusters[:, 0].min() + 1
            c1 = clusters[:, 1] - clusters[:, 1].min() + 1
            c01 = (c0 * (c1.max() + 1) + c1).astype(int64)
            s01 = cov_cluster(xhat_e, c01.squeeze())
            nc = len(unique(c01))
            s01 = rescale(s01, nc, nobs)

            s = s0 + s1 - s01

        return s

    @property
    def config(self) -> dict[str, Any]:
        """Return the options of the covariance estimator as a dict."""
        return {
            "debiased": self.debiased,
            "clusters": self._clusters,
            "kappa": self._kappa,
        }
