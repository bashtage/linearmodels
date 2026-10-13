"""
The coefficient of determination of the models when the total sum of squares is zero.
"""

from __future__ import annotations

import numpy as np

__all__ = ["r_squared", "total_ss_is_zero"]

_EPS = float(np.finfo(np.float64).eps)


def total_ss_is_zero(total_ss: float, y_ss: float, nobs: int) -> bool:
    """
    Check if a total sum of squares is zero, up to rounding error.

    Parameters
    ----------
    total_ss : float
        The total sum of squares, which is centered if the model has a constant.
    y_ss : float
        The sum of squares of the dependent variable, with the weights of the model and
        without removing a mean or the effects. It sets the size of the rounding error
        in ``total_ss``. For a total sum of squares that is not centered it is the
        total sum of squares.
    nobs : int
        The number of observations that ``total_ss`` adds.

    Returns
    -------
    bool
        True if ``total_ss`` is no larger than ``(max(nobs, 100) * eps) ** 2 * y_ss``,
        where ``eps`` is the machine precision.

    Notes
    -----
    A dependent variable that does not vary has a total sum of squares of 0, or a
    positive number that is only rounding error, e.g., the mean of a variable that is
    always 0.1 is not exactly 0.1. The sum of n numbers is off by at most about
    ``n * eps`` times the sum of their absolute values, however it is computed. The mean
    of n copies of c can then differ from c by ``n * eps * |c|``, so that every deviation
    from the mean has that size, and the sum of their squares is
    ``(n * eps) ** 2 * y_ss``, since ``y_ss`` is ``n * c ** 2``. A total sum of squares
    that is no larger is rounding error. Effects, quasi-differences and weights have the
    same bound. The rounding error that was observed in the models was smaller, at most
    0.007 ``n * eps`` for a variable that is constant, with 2 million observations in a
    model with effects, which sum the data one after the other, and a few ``eps`` with
    fewer than 1,000 observations or when the sums are computed in pairs. At least 100 is
    used for ``n`` since the error does not become smaller than that of a few additions.

    A variable that varies in the data has a total sum of squares that is larger than
    the limit unless its standard deviation is smaller than ``n * eps`` times its root
    mean square, which is 2.2e-10 with a million observations, and which cannot be
    told from rounding error in the sum.
    """
    return bool(total_ss <= (max(nobs, 100) * _EPS) ** 2 * y_ss)


def r_squared(residual_ss: float, total_ss: float, y_ss: float, nobs: int) -> float:
    """
    Compute the coefficient of determination, which is NaN without a total sum of squares.

    Parameters
    ----------
    residual_ss : float
        The residual sum of squares.
    total_ss : float
        The total sum of squares, which is centered if the model has a constant.
    y_ss : float
        The sum of squares of the dependent variable that sets the size of the rounding
        error in ``total_ss``. See `total_ss_is_zero`.
    nobs : int
        The number of observations that ``total_ss`` adds.

    Returns
    -------
    float
        ``1 - residual_ss / total_ss``, or NaN if ``total_ss`` is zero, up to rounding
        error. No error or warning is raised for a total sum of squares of 0.
    """
    if total_ss_is_zero(total_ss, y_ss, nobs):
        return float("nan")
    return float(1 - residual_ss / total_ss)
