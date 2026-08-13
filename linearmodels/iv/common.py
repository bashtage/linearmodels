from __future__ import annotations

from numpy import ix_, ptp, squeeze, where
from numpy.linalg import inv

from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
import linearmodels.typing.data
from linearmodels.iv._utility import annihilate, proj


def find_constant(x: linearmodels.typing.data.Float64Array) -> int | None:
    """
    Parameters
    ----------
    x : ndarray
        2-d array (nobs, nvar)

    Returns
    -------
    const_loc : {int, None}
        Integer location or None, if there is no constant
    """
    loc = where(ptp(x, 0) == 0)[0]
    if loc.shape != (0,):
        return loc[0]
    else:
        return None


def cragg_donald(
    endog: linearmodels.typing.data.Float64Array,
    instr: linearmodels.typing.data.Float64Array,
    exog: linearmodels.typing.data.Float64Array,
) -> WaldTestStatistic | InvalidTestStatistic:
    r"""
    Cragg-Donald test of reduced rank for the first-stage regression

    Parameters
    ----------
    endog : ndarray
        Weighted endogenous regressor array (nobs, nendog)
    instr : ndarray
        Weighted instrument array (nobs, ninstr)
    exog : ndarray
        Weighted exogenous regressor array (nobs, nexog), partialled out
        before testing. Include a constant column here if the model has one.

    Returns
    -------
    WaldTestStatistic
        Test statistic, distributed chi2(ninstr - nendog + 1) under the
        null that the first-stage coefficient matrix does not have full
        column rank. Returns an InvalidTestStatistic if there are fewer
        instruments than endogenous regressors.

    Notes
    -----
    Let :math:`X = Z\Pi + V`, where :math:`\Pi \in \mathbb{R}^{k \times m}`.
    The null hypothesis is :math:`\mathrm{rank}(\Pi) < m`. The test
    statistic is

    .. math::

        \mathrm{CD} = (n - k - c) \cdot \lambda_{\min}\left(
        (X^T M_Z X)^{-1} X^T P_Z X\right)

    where :math:`P_Z` is the projection onto the column space of the
    (control-partialled) instruments, :math:`M_Z = I - P_Z`, and
    :math:`\lambda_{\min}` is the smallest eigenvalue. With a single
    endogenous regressor, this statistic reduces to the standard
    first-stage F-statistic.

    The reported p-value uses the asymptotic chi2 distribution (Anderson
    1951) rather than Stock-Yogo (2005) finite-sample critical values,
    which require choosing a tolerance for maximal size distortion or
    worst-case bias and so are not implemented here.

    References
    ----------
    .. [1] Cragg, J. G., & Donald, S. G. (1993). Testing identifiability
       and specification in instrumental variable models. Econometric
       Theory, 9(2), 222-240.
    .. [2] Anderson, T. W. (1951). Estimating linear restrictions on
       regression coefficients for multivariate normal distributions.
       Annals of Mathematical Statistics, 22(3), 327-351.
    """
    import scipy.linalg

    n, k = instr.shape
    m = endog.shape[1]
    name = "Cragg-Donald Test"
    if m == 0:
        return InvalidTestStatistic(
            "Model contains no endogenous regressors; the Cragg-Donald "
            "statistic is not defined.",
            name=name,
        )
    if k < m:
        return InvalidTestStatistic(
            "Number of instruments is less than the number of endogenous "
            "regressors; the Cragg-Donald statistic is not defined.",
            name=name,
        )

    n_controls = exog.shape[1]
    x = annihilate(endog, exog) if n_controls > 0 else endog
    z = annihilate(instr, exog) if n_controls > 0 else instr

    x_proj = proj(x, z)
    signal = x.T @ x_proj
    noise = x.T @ (x - x_proj)

    lambda_min = scipy.linalg.eigh(signal, noise, eigvals_only=True)[0]
    statistic = (n - k - n_controls) * lambda_min

    df = k - m + 1
    null = "Instruments jointly identify the endogenous regressors (full rank)"
    return WaldTestStatistic(statistic, null, df, name=name)


def f_statistic(
    params: linearmodels.typing.data.Float64Array,
    cov: linearmodels.typing.data.Float64Array,
    debiased: bool,
    resid_df: int,
    const_loc: int | None = None,
) -> WaldTestStatistic | InvalidTestStatistic:
    """
    Parameters
    ----------
    params : ndarray
        Estimated parameters (nvar, 1)
    cov : ndarray
        Covariance of estimated parameters (nvar, nvar)
    debiased : bool
        False indicating whether to use a small-sample exact F or the large
        sample chi2 distribution
    resid_df : int
        NUmber of observations minus number of model parameters
    const_loc : int
        Location of constant column, if any

    Returns
    -------
    WaldTestStatistic
        WaldTestStatistic instance
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
    test_stat = float(squeeze(test_params.T @ inv(test_cov) @ test_params))
    df = test_params.shape[0]
    if debiased:
        wald = WaldTestStatistic(test_stat / df, null, df, resid_df, name=name)
    else:
        wald = WaldTestStatistic(test_stat, null, df, name=name)

    return wald
