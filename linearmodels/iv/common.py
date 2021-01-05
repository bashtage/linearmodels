from typing import Optional, Union

from numpy import ix_, ptp, where
from numpy.linalg import inv

from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.typing import NDArray


def find_constant(x: NDArray) -> Optional[int]:
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


def f_statistic(
    params: NDArray,
    cov: NDArray,
    debiased: bool,
    resid_df: int,
    const_loc: Optional[int] = None,
) -> Union[WaldTestStatistic, InvalidTestStatistic]:
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
    const_loc : int, optional
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
    test_stat = test_params.T @ inv(test_cov) @ test_params
    test_stat = float(test_stat)
    df = test_params.shape[0]
    if debiased:
        wald = WaldTestStatistic(test_stat / df, null, df, resid_df, name=name)
    else:
        wald = WaldTestStatistic(test_stat, null, df, name=name)

    return wald
