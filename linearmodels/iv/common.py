"""
Rank tests of the first stage and the model F-statistic.
"""

from __future__ import annotations

from typing import Any, NamedTuple
import warnings

from numpy import (
    asarray,
    column_stack,
    eye,
    finfo,
    ix_,
    kron,
    nan,
    ones,
    ptp,
    squeeze,
    unique,
    where,
)
from numpy.linalg import LinAlgError, eigh, inv, matrix_rank, svd
from scipy.linalg import cholesky, solve_triangular

from linearmodels.iv._utility import annihilate
from linearmodels.iv.gmm import (
    HeteroskedasticWeightMatrix,
    KernelWeightMatrix,
    OneWayClusteredWeightMatrix,
)
from linearmodels.shared.exceptions import unknown_option_error
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
import linearmodels.typing.data

__all__ = ["cragg_donald", "cragg_donald_f", "kleibergen_paap", "kleibergen_paap_f"]

# The null of the rank tests: the first-stage coefficient matrix is rank
# deficient, which means that the model is not identified.
_RANK_NULL = (
    "The first-stage coefficient matrix does not have full column rank "
    "(the model is underidentified)"
)
# 1 - r**2 below this is treated as an exact first stage
_EXACT_FIT_TOL = 64 * finfo(float).eps
# Names that IV models accept for each family of covariance estimators
_COV_FAMILIES = {
    "unadjusted": ("unadjusted", "homoskedastic", "homoskedasticcovariance", "homo"),
    "robust": ("robust", "heteroskedastic", "heteroskedasticcovariance", "hccm"),
    "clustered": ("clustered", "one-way", "onewayclusteredcovariance"),
    "kernel": ("kernel", "kernelcovariance"),
}


def _cov_family(cov_type: str) -> str:
    """Map the name of a covariance estimator to its family."""
    name = str(cov_type).lower()
    for family, aliases in _COV_FAMILIES.items():
        if name in aliases:
            return family
    supported = [alias for aliases in _COV_FAMILIES.values() for alias in aliases]
    raise unknown_option_error("cov_type", cov_type, supported)


def find_constant(x: linearmodels.typing.data.Float64Array) -> int | None:
    """
    Locate the first constant column of an array.

    Parameters
    ----------
    x : ndarray
        2-d array (nobs, nvar).

    Returns
    -------
    int or None
        Integer location of the first constant column, or None if there is
        no constant.
    """
    loc = where(ptp(x, 0) == 0)[0]
    if loc.shape != (0,):
        return int(loc[0])
    else:
        return None


class _FirstStageRank(NamedTuple):
    """Canonical correlation decomposition of the partialled first stage."""

    nobs: int
    ninstr: int
    nendog: int
    nexog: int
    x: linearmodels.typing.data.Float64Array
    z: linearmodels.typing.data.Float64Array
    pihat: linearmodels.typing.data.Float64Array
    theta: linearmodels.typing.data.Float64Array
    u: linearmodels.typing.data.Float64Array
    s: linearmodels.typing.data.Float64Array
    v: linearmodels.typing.data.Float64Array
    irqzz: linearmodels.typing.data.Float64Array
    irqxx: linearmodels.typing.data.Float64Array


def _first_stage_rank(
    endog: linearmodels.typing.data.Float64Array,
    instr: linearmodels.typing.data.Float64Array,
    exog: linearmodels.typing.data.Float64Array,
    name: str,
) -> _FirstStageRank | InvalidTestStatistic:
    r"""
    Decompose the first stage into canonical correlations.

    The endogenous regressors and excluded instruments are first
    orthogonalized with respect to the exogenous regressors. Let
    :math:`Q_{zz}`, :math:`Q_{xx}` and :math:`Q_{zx}` be the second moments
    of the orthogonalized instruments and regressors. The matrix

    .. math::

        \hat{\Theta} = Q_{zz}^{-1/2\prime} Q_{zx} Q_{xx}^{-1/2}

    is formed using Cholesky factors, and its singular values are the sample
    partial canonical correlations. Both rank tests are functions of this
    decomposition.

    Returns an InvalidTestStatistic that explains why the tests are not
    defined if the decomposition cannot be computed.
    """
    nobs, ninstr = instr.shape
    nendog = endog.shape[1]
    nexog = exog.shape[1]
    if nendog == 0:
        return InvalidTestStatistic(
            f"Model contains no endogenous regressors; the {name} statistic "
            "is not defined.",
            name=name,
        )
    if ninstr < nendog:
        return InvalidTestStatistic(
            "Number of instruments is less than the number of endogenous "
            f"regressors; the {name} statistic is not defined.",
            name=name,
        )
    x = asarray(endog, dtype=float)
    z = asarray(instr, dtype=float)
    if nexog > 0:
        x = annihilate(x, exog)
        z = annihilate(z, exog)
    if matrix_rank(x) < nendog:
        return InvalidTestStatistic(
            "The endogenous regressors are collinear once the exogenous "
            f"regressors are partialled out; the {name} statistic is not "
            "defined.",
            name=name,
        )
    if matrix_rank(z) < ninstr:
        return InvalidTestStatistic(
            "The instruments are collinear once the exogenous regressors are "
            f"partialled out; the {name} statistic is not defined.",
            name=name,
        )
    qzz = z.T @ z / nobs
    qxx = x.T @ x / nobs
    qzx = z.T @ x / nobs
    # Upper triangular Cholesky factors with R'R = Q
    irqzz = solve_triangular(cholesky(qzz), eye(ninstr))
    irqxx = solve_triangular(cholesky(qxx), eye(nendog))
    pihat = irqzz @ (irqzz.T @ qzx)
    theta = irqzz.T @ qzx @ irqxx
    u, s, vt = svd(theta, full_matrices=True)
    if 1.0 - float(s[-1]) ** 2 <= _EXACT_FIT_TOL:
        return InvalidTestStatistic(
            "The endogenous regressors are exactly predicted by the "
            f"instruments, so the {name} statistic is not defined.",
            name=name,
        )
    return _FirstStageRank(
        nobs, ninstr, nendog, nexog, x, z, pihat, theta, u, s, vt.T, irqzz, irqxx
    )


def cragg_donald(
    endog: linearmodels.typing.data.Float64Array,
    instr: linearmodels.typing.data.Float64Array,
    exog: linearmodels.typing.data.Float64Array,
) -> WaldTestStatistic | InvalidTestStatistic:
    r"""
    Cragg-Donald test of reduced rank for the first-stage regression.

    Parameters
    ----------
    endog : ndarray
        Weighted endogenous regressor array (nobs, nendog).
    instr : ndarray
        Weighted instrument array (nobs, ninstr).
    exog : ndarray
        Weighted exogenous regressor array (nobs, nexog), partialled out
        before testing. Include a constant column here if the model has one.

    Returns
    -------
    WaldTestStatistic or InvalidTestStatistic
        Test statistic, distributed chi2(ninstr - nendog + 1) under the null
        that the first-stage coefficient matrix does not have full column
        rank, which is the null that the model is underidentified. An
        InvalidTestStatistic is returned if the statistic is not defined: when
        there are no endogenous regressors, fewer instruments than endogenous
        regressors, collinear endogenous regressors or instruments, or when
        the instruments predict the endogenous regressors exactly.

    Warnings
    --------
    The statistic is valid only when the first-stage errors are
    conditionally homoskedastic and serially uncorrelated. It ignores the
    covariance estimator used to fit the model. If the data have
    heteroskedasticity, within-group dependence or serial correlation, it can
    reject the null of underidentification much too often. The distortion is
    modest with heteroskedasticity alone, but it can be very large when the
    instruments and the first-stage errors are both clustered or persistent.
    Use :func:`kleibergen_paap` with a covariance estimator that matches the
    dependence in the data in those cases.

    The null hypothesis is that the model is underidentified. Rejecting it
    shows that the instruments are not exactly uninformative. It does not
    show that they are strong. With weak instruments the asymptotic
    :math:`\chi^2` approximation is poor, and 2SLS can be badly biased and
    its tests badly sized even when the null is rejected.

    The p-value is asymptotic. Stock and Yogo (2005) critical values for weak
    instruments apply to the F form under i.i.d. errors and are not
    provided here. Choosing whether to use an instrumental variable
    estimator based on the outcome of a test like this one distorts the
    inference that follows.

    See Also
    --------
    cragg_donald_f
        The F form of the statistic used with Stock-Yogo critical values.
    kleibergen_paap
        Generalization that does not require homoskedastic errors.

    Notes
    -----
    Let :math:`X = Z\Pi + V`, where :math:`\Pi` is the :math:`k \times m`
    matrix of first-stage coefficients on the excluded instruments, after
    the exogenous regressors have been partialled out. The null hypothesis
    is :math:`\mathrm{rank}(\Pi) < m`. The statistic is the minimum
    eigenvalue statistic of Cragg and Donald (1993),

    .. math::

        \mathrm{CD} = (n - k - c) \frac{r^2_{\min}}{1 - r^2_{\min}}

    where :math:`r_{\min}` is the smallest sample partial canonical
    correlation between the endogenous regressors and the excluded
    instruments, and :math:`c` is the number of exogenous regressors.
    :math:`r^2_{\min}/(1 - r^2_{\min})` is the smallest eigenvalue of
    :math:`(X'M_ZX)^{-1}X'P_ZX`. The statistic is asymptotically
    :math:`\chi^2_{k-m+1}` under the null. Anderson (1951) derived a test of
    the rank that is a function of the canonical correlations.

    The statistic is **not** the first-stage F-statistic. It equals :math:`k`
    times the classical F-statistic when there is a single endogenous
    regressor. The F form, ``CD / k``, is reported by :func:`cragg_donald_f`.

    **Weights.** In a weighted model, each of the arrays has to be multiplied
    by the square root of the observation weights, including the constant
    column of ``exog``. The constant is then a column of the square roots of
    the weights and not a column of ones, since the first-stage regressions
    are weighted. The sample size :math:`n` is the number of rows, and not the
    sum of the weights. The statistic does not change if all of the weights are
    multiplied by the same constant. The results of the IV models do this
    using weights that have been scaled to have mean 1.

    **Clusters.** The statistic does not use a covariance estimator, and so
    does not depend on clusters. It is the same for a model that is estimated
    with a clustered, kernel or robust covariance estimator as it is for one
    with an unadjusted covariance estimator. See :func:`kleibergen_paap` for a
    statistic that does.

    References
    ----------
    Cragg, J. G., & Donald, S. G. (1993). Testing identifiability and
    specification in instrumental variable models. *Econometric Theory*, 9(2),
    222-240.

    Anderson, T. W. (1951). Estimating linear restrictions on regression
    coefficients for multivariate normal distributions. *Annals of
    Mathematical Statistics*, 22(3), 327-351.

    Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV
    regression. In *Identification and Inference for Econometric Models*.
    Cambridge University Press.
    """
    name = "Cragg-Donald Test"
    dec = _first_stage_rank(endog, instr, exog, name)
    if isinstance(dec, InvalidTestStatistic):
        return dec
    r2 = float(dec.s[-1]) ** 2
    statistic = (dec.nobs - dec.ninstr - dec.nexog) * r2 / (1.0 - r2)
    df = dec.ninstr - dec.nendog + 1
    return WaldTestStatistic(statistic, _RANK_NULL, df, name=name)


def cragg_donald_f(
    endog: linearmodels.typing.data.Float64Array,
    instr: linearmodels.typing.data.Float64Array,
    exog: linearmodels.typing.data.Float64Array,
) -> float:
    r"""
    Cragg-Donald Wald F statistic.

    Parameters
    ----------
    endog : ndarray
        Weighted endogenous regressor array (nobs, nendog).
    instr : ndarray
        Weighted instrument array (nobs, ninstr).
    exog : ndarray
        Weighted exogenous regressor array (nobs, nexog), partialled out
        before testing. Include a constant column here if the model has one.

    Returns
    -------
    float
        The Cragg-Donald statistic divided by the number of excluded
        instruments, or NaN if the statistic is not defined. See
        :func:`cragg_donald`.

    Warnings
    --------
    Stock-Yogo critical values are valid only for conditionally
    homoskedastic, serially uncorrelated errors, and the statistic is not a
    test: it has no p-value. A value above a threshold does not guarantee
    that the instruments are strong enough for reliable inference. See the
    warnings in :func:`cragg_donald`.

    Notes
    -----
    .. math::

        F_{CD} = \frac{n - k - c}{k} \frac{r^2_{\min}}{1 - r^2_{\min}}

    With a single endogenous regressor this is the classical F-statistic for
    the joint significance of the excluded instruments in the first-stage
    regression. It is the statistic that Stock and Yogo (2005) tabulate
    critical values for. It is reported by Stata's ``ivreg2`` as the
    "Cragg-Donald Wald F statistic".

    The weights are used as in :func:`cragg_donald`. Each array, including the
    constant of ``exog``, is multiplied by the square root of the weights in a
    weighted model, and the statistic does not depend on the scale of the
    weights or on clusters.

    References
    ----------
    Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV
    regression. In *Identification and Inference for Econometric Models*.
    Cambridge University Press.
    """
    test = cragg_donald(endog, instr, exog)
    if isinstance(test, InvalidTestStatistic):
        return nan
    return float(test.stat) / instr.shape[1]


def _score_covariance(
    v: linearmodels.typing.data.Float64Array,
    z: linearmodels.typing.data.Float64Array,
    family: str,
    cov_config: dict[str, Any],
) -> linearmodels.typing.data.Float64Array:
    """
    Covariance of the first-stage scores vec(Z'V) / sqrt(n)

    The scores are ordered by endogenous regressor and then by instrument,
    which is the order of the column-stacked first-stage coefficient matrix.
    ``family`` must be one of "robust", "clustered" or "kernel". Clusters
    must be a 1-d array.
    """
    nobs = z.shape[0]
    scores = column_stack([v[:, [j]] * z for j in range(v.shape[1])])
    eps = ones((nobs, 1))
    estimator: (
        HeteroskedasticWeightMatrix | OneWayClusteredWeightMatrix | KernelWeightMatrix
    )
    if family == "robust":
        estimator = HeteroskedasticWeightMatrix(center=False, debiased=False)
    elif family == "clustered":
        estimator = OneWayClusteredWeightMatrix(
            cov_config["clusters"], center=False, debiased=False
        )
    else:
        estimator = KernelWeightMatrix(
            kernel=str(cov_config.get("kernel", "bartlett")),
            bandwidth=cov_config.get("bandwidth"),
            center=False,
            debiased=False,
            optimal_bw=True,
        )
    return estimator.weight_matrix(scores, scores, eps)


def _rk_chi2(
    dec: _FirstStageRank,
    v: linearmodels.typing.data.Float64Array,
    family: str,
    cov_config: dict[str, Any],
) -> tuple[float, int] | None:
    """
    Kleibergen-Paap rk statistic for the null that rank(Pi) = nendog - 1

    ``v`` contains the series that are interacted with the instruments to
    form the scores. Returns the chi2 statistic and its degrees of freedom,
    or None if the covariance of the statistic is zero.
    """
    shat = _score_covariance(v, dec.z, family, cov_config)
    # Covariance of vec(Theta) from the covariance of the scores
    transform = kron(dec.irqxx.T, dec.irqzz.T)
    kpvar = transform @ shat @ transform.T
    # Bases of the left and right singular spaces for the smallest singular
    # values of Theta. Any basis of the same spaces gives the same statistic.
    kk = dec.nendog - 1
    left = dec.u[:, kk:]
    right = dec.v[:, kk:]
    select = kron(right.T, left.T)
    lam = select @ dec.theta.ravel(order="F")
    vlam = select @ kpvar @ select.T
    vlam = (vlam + vlam.T) / 2
    values, vectors = eigh(vlam)
    tol = values.max() * finfo(float).eps * vlam.shape[0]
    positive = values > tol
    if not positive.any():
        return None
    deficit = int((~positive).sum())
    if deficit:
        warnings.warn(
            "The covariance of the Kleibergen-Paap statistic is rank deficient "
            f"(rank deficit = {deficit}). This is usually caused by too few "
            "clusters or observations relative to the number of instruments. "
            "A generalized inverse is used and the degrees of freedom are "
            "reduced. The statistic should not be relied upon.",
            UserWarning,
            stacklevel=4,
        )
    coef = vectors[:, positive].T @ lam
    chi2 = dec.nobs * float((coef**2 / values[positive]).sum())
    return chi2, int(vlam.shape[0] - deficit)


def _kleibergen_paap(
    endog: linearmodels.typing.data.Float64Array,
    instr: linearmodels.typing.data.Float64Array,
    exog: linearmodels.typing.data.Float64Array,
    cov_type: str,
    cov_config: dict[str, Any] | None,
    *,
    wald: bool = True,
) -> tuple[WaldTestStatistic | InvalidTestStatistic, float]:
    """
    Compute the rk LM test and the rk Wald F-statistic

    The F-statistic is NaN if wald is False. The covariance of the scores of
    the Wald statistic is not computed in that case. Its rank can be less than
    the rank of the covariance of the LM statistic, and so it can warn when
    the LM statistic is well-defined.
    """
    name = "Kleibergen-Paap rk LM Test"
    family = _cov_family(cov_type)
    config = {} if cov_config is None else dict(cov_config)
    if family == "clustered":
        if config.get("clusters") is None:
            raise ValueError("clusters are required when cov_type is 'clustered'.")
        clusters = asarray(config["clusters"])
        if clusters.ndim == 2 and clusters.shape[1] == 1:
            clusters = clusters[:, 0]
        if clusters.ndim != 1:
            reason = "The statistic is not available for two-way clustering."
            return InvalidTestStatistic(reason, name=name), nan
        if unique(clusters).shape[0] < 2:
            reason = "The statistic requires at least two clusters."
            return InvalidTestStatistic(reason, name=name), nan
        config["clusters"] = clusters
    dec = _first_stage_rank(endog, instr, exog, name)
    if isinstance(dec, InvalidTestStatistic):
        return dec, nan
    df = dec.ninstr - dec.nendog + 1
    r2 = float(dec.s[-1]) ** 2
    if family == "unadjusted":
        # With a Kronecker covariance the rk statistics reduce to Anderson's
        # canonical correlation LM statistic and the Cragg-Donald F
        lm = dec.nobs * r2
        f = (dec.nobs - dec.ninstr - dec.nexog) * r2 / (1.0 - r2) / dec.ninstr
        return WaldTestStatistic(lm, _RANK_NULL, df, name=name), f

    singular = InvalidTestStatistic(
        "The covariance of the statistic is singular, so the statistic is "
        "not defined.",
        name=name,
    )
    lm_result = _rk_chi2(dec, dec.x, family, config)
    if lm_result is None:
        return singular, nan
    lm_chi2, lm_df = lm_result
    lm_test = WaldTestStatistic(lm_chi2, _RANK_NULL, lm_df, name=name)
    f = nan
    if wald:
        # The Wald version uses the first-stage residuals to form the scores
        residuals = dec.x - dec.z @ dec.pihat
        wald_result = _rk_chi2(dec, residuals, family, config)
        dof = dec.nobs - dec.ninstr - dec.nexog
        if family == "clustered":
            nclusters = unique(config["clusters"]).shape[0]
            scale = dof * (nclusters - 1) / nclusters / (dec.nobs - 1)
        else:
            scale = dof / dec.nobs
        if wald_result is not None:
            f = wald_result[0] * scale / dec.ninstr
    return lm_test, f


def kleibergen_paap(
    endog: linearmodels.typing.data.Float64Array,
    instr: linearmodels.typing.data.Float64Array,
    exog: linearmodels.typing.data.Float64Array,
    cov_type: str = "robust",
    cov_config: dict[str, Any] | None = None,
) -> WaldTestStatistic | InvalidTestStatistic:
    r"""
    Kleibergen-Paap rk LM test of underidentification.

    Parameters
    ----------
    endog : ndarray
        Weighted endogenous regressor array (nobs, nendog).
    instr : ndarray
        Weighted instrument array (nobs, ninstr).
    exog : ndarray
        Weighted exogenous regressor array (nobs, nexog), partialled out
        before testing. Include a constant column here if the model has one.
    cov_type : str, optional
        Covariance estimator used for the first-stage scores. The default is
        "robust". The names are not case sensitive. Supported covariance
        estimators are:

        * "unadjusted", "homoskedastic", "homo", "homoskedasticcovariance" -
          Classic homoskedastic inference
        * "robust", "heteroskedastic", "hccm", "heteroskedasticcovariance" -
          Heteroskedasticity robust inference
        * "kernel", "kernelcovariance" - Heteroskedasticity and autocorrelation
          robust inference
        * "clustered", "one-way", "onewayclusteredcovariance" - Cluster
          dependent inference, which is heteroskedasticity robust

    cov_config : dict, optional
        Options for the covariance estimator. "clustered" requires
        ``clusters``, a 1-d array with one element for each observation.
        "kernel" uses ``kernel`` (default "bartlett") and ``bandwidth``. The
        names of the kernels are not case sensitive. If the bandwidth is not
        provided it is selected automatically, which needs at least 4
        observations with the Bartlett kernel, 5 with the Parzen kernel and 6
        with the quadratic spectral kernel. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`, which
        lists the supported kernels.

    Returns
    -------
    WaldTestStatistic or InvalidTestStatistic
        Test statistic, distributed chi2(ninstr - nendog + 1) under the null
        that the first-stage coefficient matrix does not have full column
        rank, which is the null that the model is underidentified. An
        InvalidTestStatistic is returned if the statistic is not defined.
        See :func:`cragg_donald`. It is also returned for two-way clustering,
        which is not supported, for fewer than two clusters, and if the
        covariance of the statistic is singular.

    Raises
    ------
    ValueError
        If ``cov_type`` is not supported, or is "clustered" and
        ``cov_config`` does not contain ``clusters`` or ``clusters`` does not
        have an element for each row of the arrays. It is also raised if
        ``cov_type`` is "kernel", the bandwidth is not provided and there are
        fewer observations than the kernel needs to select it, or the
        bandwidth that is selected is not finite.

    Warnings
    --------
    The statistic is asymptotic. It is robust only to the extent that the
    chosen covariance estimator is appropriate for the data. Use "clustered"
    when observations are dependent within groups, and "kernel" when the
    data are ordered in time and the scores are autocorrelated. The data must
    be sorted in time order when "kernel" is used. If the dependence is not
    accounted for, including by "unadjusted" and "robust", the statistic can
    reject the null of underidentification much too often when the
    instruments and first-stage errors are clustered or persistent.

    The covariance has dimension :math:`km \times km` and is poorly estimated
    unless :math:`n` is large relative to :math:`km`. It is formed as a dense
    matrix, and so the memory that is needed grows with :math:`(km)^2`. With
    clustering it needs many clusters, and with few clusters the test can be
    very conservative. A kernel bandwidth that is too short for the dependence
    does not remove the over-rejection. If there are too few clusters or
    observations the covariance is rank deficient. A warning is issued, a
    generalized inverse is used and the degrees of freedom are reduced, but
    the result should not be relied on. The chi-square approximation is also
    less reliable with many instruments.

    The null is that the model is underidentified. Rejecting it does not
    show that the instruments are strong, so this is not a test for weak
    instruments. With weak instruments the asymptotic distribution can be a
    poor approximation. Choosing whether to use an instrumental variable
    estimator based on the outcome of a test like this one distorts the
    inference that follows.

    See Also
    --------
    kleibergen_paap_f
        The F form of the statistic.
    cragg_donald
        The version that requires homoskedastic errors.

    Notes
    -----
    Let :math:`\hat{\Pi}` be the :math:`k \times m` first-stage coefficient
    matrix after the exogenous regressors have been partialled out. The test
    is applied to :math:`\hat{\Theta}`, the matrix of sample partial
    canonical correlations between the endogenous regressors and the
    excluded instruments. The null is that :math:`\Theta` has rank :math:`m-1`.
    Let :math:`\hat{\Theta} = USV'` be a singular value decomposition, and
    let :math:`A_\perp` and :math:`B_\perp` contain the singular vectors of
    the smallest :math:`m-1` and :math:`1` singular values. The statistic is

    .. math::

        rk = n \,\mathrm{vec}(\lambda)'
        \left[(B_\perp \otimes A_\perp)' \hat{V}_{\Theta}
        (B_\perp \otimes A_\perp)\right]^{-1} \mathrm{vec}(\lambda),
        \quad \lambda = A_\perp' \hat{\Theta} B_\perp

    where :math:`\hat{V}_\Theta` is the estimated covariance of
    :math:`\sqrt{n}\,\mathrm{vec}(\hat{\Theta})`. It is built from the scores
    :math:`x_i \otimes z_i`, where :math:`x_i` is the orthogonalized
    endogenous regressor and :math:`z_i` is the orthogonalized instrument,
    using the covariance estimator chosen by ``cov_type``. The statistic is
    asymptotically :math:`\chi^2_{k-m+1}` under the null.

    With ``cov_type`` "unadjusted" the covariance has a Kronecker structure
    and the statistic is Anderson's canonical correlation LM statistic,
    :math:`n r^2_{\min}`. No small-sample adjustment is applied to the
    covariance, so the ``debiased`` option of the IV models has no effect on
    these statistics.

    This is the statistic that Stata's ``ivreg2`` reports as the
    "Kleibergen-Paap rk LM statistic" in its underidentification test. It is
    computed following ``ranktest`` (Kleibergen and Schaffer). Results can
    differ in detail from Stata for kernel covariances since the bandwidth
    conventions differ: the Bartlett weight on lag :math:`j` here is
    :math:`1 - j/(b+1)`, so a Stata bandwidth of :math:`b+1` corresponds to a
    ``bandwidth`` of :math:`b`.

    **Weights.** In a weighted model, each of the arrays has to be multiplied
    by the square root of the observation weights, including the constant
    column of ``exog``, as for :func:`cragg_donald`. The scores are then those
    of the weighted first-stage regressions, and the covariance estimator is
    applied to them. The statistic does not change if all of the weights are
    multiplied by the same constant.

    **Clusters.** With "clustered", the scores are summed within the clusters
    in ``cov_config["clusters"]``. ``clusters`` has one element for each row
    of the arrays, in the order of the rows. It is a 1-d array, or a 2-d array
    with a single column. Two-way clustering is not supported. The LM
    statistic is a quadratic form in a covariance with :math:`k-m+1` rows,
    whose rank is at most the number of clusters, and so it needs at least
    :math:`k-m+1` clusters. The scores of the Wald F statistic of
    :func:`kleibergen_paap_f` sum to zero, and so it needs one more cluster.
    A warning is issued if there are fewer clusters.

    References
    ----------
    Kleibergen, F., & Paap, R. (2006). Generalized reduced rank tests using the
    singular value decomposition. *Journal of Econometrics*, 133(1), 97-126.

    Kleibergen, F., & Schaffer, M. E. (2007). ranktest: Stata module to test
    the rank of a matrix using the Kleibergen-Paap rk statistic. *Statistical
    Software Components* S456865, Boston College.
    """
    return _kleibergen_paap(endog, instr, exog, cov_type, cov_config, wald=False)[0]


def kleibergen_paap_f(
    endog: linearmodels.typing.data.Float64Array,
    instr: linearmodels.typing.data.Float64Array,
    exog: linearmodels.typing.data.Float64Array,
    cov_type: str = "robust",
    cov_config: dict[str, Any] | None = None,
) -> float:
    r"""
    Kleibergen-Paap rk Wald F statistic.

    Parameters
    ----------
    endog : ndarray
        Weighted endogenous regressor array (nobs, nendog).
    instr : ndarray
        Weighted instrument array (nobs, ninstr).
    exog : ndarray
        Weighted exogenous regressor array (nobs, nexog), partialled out
        before testing. Include a constant column here if the model has one.
    cov_type : str, optional
        Covariance estimator used for the first-stage scores. The default is
        "robust". The names are not case sensitive. Supported covariance
        estimators are:

        * "unadjusted", "homoskedastic", "homo", "homoskedasticcovariance" -
          Classic homoskedastic inference
        * "robust", "heteroskedastic", "hccm", "heteroskedasticcovariance" -
          Heteroskedasticity robust inference
        * "kernel", "kernelcovariance" - Heteroskedasticity and autocorrelation
          robust inference
        * "clustered", "one-way", "onewayclusteredcovariance" - Cluster
          dependent inference, which is heteroskedasticity robust

    cov_config : dict, optional
        Options for the covariance estimator. See :func:`kleibergen_paap`.

    Returns
    -------
    float
        The rk Wald F statistic, or NaN if it is not defined.

    Raises
    ------
    ValueError
        If ``cov_type`` is not supported, or is "clustered" and
        ``cov_config`` does not contain ``clusters``. It is also raised if
        ``cov_type`` is "kernel", the bandwidth is not provided and there are
        fewer observations than the kernel needs to select it, or the
        bandwidth that is selected is not finite. See :func:`kleibergen_paap`.

    Warnings
    --------
    The statistic is a robust analogue of the Cragg-Donald F statistic, but
    it is **not** a test and has no p-value. Stock-Yogo critical values are
    derived for conditionally homoskedastic, serially uncorrelated errors and
    do not apply to this statistic when the errors are not i.i.d.
    Comparing the two is a common heuristic without a formal justification.
    For a single endogenous regressor, the effective F-statistic of Olea and
    Pflueger (2013) is a better-founded choice but is not implemented here.
    Even then, a large first-stage F-statistic does not ensure that
    confidence intervals based on 2SLS have correct coverage.

    The warnings in :func:`kleibergen_paap` also apply: the statistic is
    asymptotic, needs :math:`n` to be large relative to the number of
    instruments times the number of endogenous regressors, and needs many
    clusters if clustered. Using the statistic to decide whether to proceed
    with IV estimation distorts subsequent inference.

    Notes
    -----
    The rk Wald statistic has the form of the statistic in
    :func:`kleibergen_paap`, but the scores are formed from the first-stage
    residuals rather than from the orthogonalized endogenous regressors. If
    :math:`W` is the statistic, the F statistic is

    .. math::

        F_{rk} = \frac{W}{n} \frac{n - k - c}{k}

    where :math:`k` is the number of excluded instruments and :math:`c` is the
    number of exogenous regressors. With clustered covariance, the
    statistic is

    .. math::

        F_{rk} = \frac{W}{n - 1} \frac{(n - k - c)(M - 1)}{M k}

    where :math:`M` is the number of clusters. With ``cov_type`` equal to
    "unadjusted" or "homoskedastic" the statistic is the Cragg-Donald F
    statistic from :func:`cragg_donald_f`. These are the definitions used by
    Stata's ``ivreg2`` for the "Kleibergen-Paap rk Wald F statistic" (Baum,
    Schaffer and Stillman, 2007).

    The weights and clusters are used as in :func:`kleibergen_paap`. In a
    weighted model, each of the arrays, including the constant of ``exog``, has
    to be multiplied by the square root of the weights, and the statistic does
    not depend on their scale. The clusters have one element for each row of
    the arrays. If there are :math:`m` endogenous regressors, the scores of the
    Wald statistic sum to zero, so that at least :math:`k-m+2` clusters are
    needed, which is one more than for :func:`kleibergen_paap`. A warning is
    issued if there are fewer clusters.

    References
    ----------
    Kleibergen, F., & Paap, R. (2006). Generalized reduced rank tests using the
    singular value decomposition. *Journal of Econometrics*, 133(1), 97-126.

    Olea, J. L. M., & Pflueger, C. (2013). A robust test for weak instruments.
    *Journal of Business & Economic Statistics*, 31(3), 358-369.

    Baum, C. F., Schaffer, M. E., & Stillman, S. (2007). Enhanced routines for
    instrumental variables/GMM estimation and testing. *Stata Journal*, 7(4),
    465-506.
    """
    return _kleibergen_paap(endog, instr, exog, cov_type, cov_config)[1]


def f_statistic(
    params: linearmodels.typing.data.Float64Array,
    cov: linearmodels.typing.data.Float64Array,
    debiased: bool,
    resid_df: int,
    const_loc: int | None = None,
) -> WaldTestStatistic | InvalidTestStatistic:
    """
    Test that all of the parameters other than the constant are zero.

    Parameters
    ----------
    params : ndarray
        Estimated parameters (nvar, 1).
    cov : ndarray
        Covariance of estimated parameters (nvar, nvar).
    debiased : bool
        Whether to use a small-sample F distribution (True) or the large
        sample chi2 distribution (False).
    resid_df : int
        Number of observations minus number of model parameters.
    const_loc : int, optional
        Location of the constant column, if any. The default is None, which
        means that the model has no constant.

    Returns
    -------
    WaldTestStatistic or InvalidTestStatistic
        The test statistic, or an InvalidTestStatistic if the model has no
        regressors other than the constant or the covariance of the
        parameters is singular.
    """
    null = "All parameters ex. constant are zero"
    name = "Model F-statistic"

    nvar = params.shape[0]
    non_const = list(range(nvar))
    if const_loc is not None:
        non_const.pop(const_loc)
    if not non_const:
        return InvalidTestStatistic(
            "Model contains no non-constant exogenous terms", name=name
        )
    test_params = params[non_const]
    test_cov = cov[ix_(non_const, non_const)]
    try:
        inv_test_cov = inv(test_cov)
    except LinAlgError:
        # The model fits exactly, e.g., the dependent variable is all zero, so that
        # the covariance of the parameters is 0
        return InvalidTestStatistic(
            "The covariance of the parameters is singular", name=name
        )
    test_stat = float(squeeze(test_params.T @ inv_test_cov @ test_params))
    df = test_params.shape[0]
    if debiased:
        wald = WaldTestStatistic(test_stat / df, null, df, resid_df, name=name)
    else:
        wald = WaldTestStatistic(test_stat, null, df, name=name)

    return wald
