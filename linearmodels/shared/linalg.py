from typing import Optional, Tuple

import numpy as np

from linearmodels.typing import NDArray


def has_constant(
    x: NDArray, x_rank: Optional[int] = None
) -> Tuple[bool, Optional[int]]:
    """
    Parameters
    ----------
    x: ndarray
        Array to be checked for a constant (n,k)
    x_rank : {int, None}
        Rank of x if previously computed.  If None, this value will be
        computed.

    Returns
    -------
    const : bool
        Flag indicating whether x contains a constant or has column span with
        a constant
    loc : int
        Column location of constant
    """
    if np.any(np.all(x == 1, axis=0)):
        loc: Optional[int] = int(np.argwhere(np.all(x == 1, axis=0)))
        return True, loc

    if np.any((np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0)):
        loc_arr = (np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0)
        loc = int(np.argwhere(loc_arr))
        return True, loc

    n = x.shape[0]
    aug_rank = np.linalg.matrix_rank(np.c_[np.ones((n, 1)), x])
    rank = np.linalg.matrix_rank(x) if x_rank is None else x_rank

    has_const = (aug_rank == rank) and x.shape[0] > x.shape[1]
    has_const = has_const or rank < min(x.shape)
    loc = None
    if has_const:
        normed_var = x.var(0) / np.abs(x).max(0)
        loc = int(np.argmin(normed_var))

    return bool(has_const), loc


def inv_sqrth(x: NDArray) -> NDArray:
    """
    Matrix inverse square root

    Parameters
    ----------
    x : ndarray
        Real, symmetric matrix

    Returns
    -------
    ndarray
        Input to the power -1/2
    """
    vals, vecs = np.linalg.eigh(x)
    return vecs @ np.diag(1 / np.sqrt(vals)) @ vecs.T
