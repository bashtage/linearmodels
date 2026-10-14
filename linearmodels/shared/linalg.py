"""
Linear algebra functions that the models share.
"""

from __future__ import annotations

import numpy as np

from linearmodels.typing import Float64Array

# The column of ones is in the column space of a matrix if its relative distance from
# the space is no larger than this. It is the square root of the machine precision,
# about 1.5e-8. See has_constant.
_CONSTANT_TOLERANCE = float(np.sqrt(np.finfo(np.float64).eps))


def has_constant(x: Float64Array, x_rank: int | None = None) -> tuple[bool, int]:
    """
    Check whether the columns of an array contain a constant or span one.

    Parameters
    ----------
    x : array_like
        Array to be checked for a constant, with shape (nobs, nvar).
    x_rank : int, optional
        Not used. The rank of x is found when the constant is searched for, and a
        value that is not the rank would change the answer. It is accepted so that
        existing calls continue to work.

    Returns
    -------
    const : bool
        Flag indicating whether x contains a constant or has column span with
        a constant.
    loc : int
        Column location of the constant, which is -1 if there is no constant. If the
        constant is not a column, the location is that of the column that varies the
        least relative to its size.

    Notes
    -----
    A constant is explicit if a column has the same value in every row. Its location
    is that of the first column of ones, and if there is none, that of the first other
    column with a single value.

    Otherwise, x has an implicit constant if the column of ones is in the column space
    of x, for example, if the columns include the dummy variables for every category
    of a variable. The test is exact for any rank of x. The columns are scaled to have
    a largest absolute value of 1, which does not change the column space, and the
    column of ones is regressed on them using a least squares solver based on the
    singular value decomposition, which ignores the directions that the columns do not
    span. The column of ones is in the column space if the norm of the residual,
    relative to the norm of the column of ones, is no larger than the square root of
    the machine precision, about 1.5e-8. This ratio is the sine of the angle between
    the column of ones and the column space. It is 0 if the column of ones is in the
    column space and is of the order of 1 for most data. The rounding error in the
    ratio is about the machine precision times the condition number of the scaled
    columns, so that the tolerance allows condition numbers up to about 1e8. A column
    that differs from a constant by less than the tolerance, relative to its size, is
    read as a constant.

    If the rank of x is the number of rows, its columns span every column, and so also
    the column of ones. This is not read as a constant.
    """
    x = np.asarray(x, dtype=np.float64)
    # With more than one constant column, the location is that of the first
    is_ones = np.all(x == 1, axis=0)
    if np.any(is_ones):
        return True, int(np.argmax(is_ones))

    is_single_valued = (np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0)
    if np.any(is_single_valued):
        return True, int(np.argmax(is_single_valued))

    nobs = x.shape[0]
    # Columns of 0 do not change the column space. Scaling the others does not change
    # it either, but it makes the singular values comparable if the columns differ in size
    peak = np.maximum(x.max(axis=0), -x.min(axis=0))
    nonzero = peak != 0
    if not np.any(nonzero):
        return False, -1
    scaled = x[:, nonzero] / peak[nonzero]
    ones = np.ones(nobs)
    cond = max(scaled.shape) * np.finfo(np.float64).eps
    coef, _, rank, _ = np.linalg.lstsq(scaled, ones, rcond=cond)
    relative_resid = np.linalg.norm(ones - scaled @ coef) / np.sqrt(nobs)
    if rank >= nobs or relative_resid > _CONSTANT_TOLERANCE:
        return False, -1

    # The column that varies the least relative to its size, not a column of zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        normed_var = x.var(0) / peak
    normed_var[~nonzero] = np.inf
    return True, int(np.argmin(normed_var))


def inv_sqrth(x: Float64Array) -> Float64Array:
    """
    Matrix inverse square root.

    Parameters
    ----------
    x : array_like
        Real, symmetric, positive definite matrix.

    Returns
    -------
    ndarray
        The input to the power -1/2.

    Raises
    ------
    numpy.linalg.LinAlgError
        If an eigenvalue of x is not positive, which means that x is singular, to
        rounding error, or not positive semi-definite. The inverse square root is not
        defined then, and the eigenvalue decomposition does not raise.
    """
    vals, vecs = np.linalg.eigh(x)
    if not np.all(vals > 0):
        raise np.linalg.LinAlgError(
            "The matrix is not positive definite, so that its inverse square root "
            "is not defined."
        )
    return vecs @ np.diag(1 / np.sqrt(vals)) @ vecs.T
