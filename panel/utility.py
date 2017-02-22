import numpy as np

from numpy import diag, sqrt
from numpy.linalg import matrix_rank, eigh


def has_constant(x):
    """
    Parameters
    ----------
    x: ndarray
        Array to be checked for a constant (n,k)
     
    Returns
    -------
    const : bool
        Flag indicating whether x contains a constant or has column span with 
        a constant
    """
    if np.any(np.all(x == 1, axis=0)):
        return True

    if np.any((np.ptp(x, axis=0) == 0) & ~np.all(x == 0, axis=0)):
        return True

    n = x.shape[0]
    aug_rank = matrix_rank(np.c_[np.ones((n, 1)), x])
    rank = matrix_rank(x)
    return aug_rank == rank


def inv_sqrth(x):
    """
    Matrix inverse square root
    
    Parameters
    ----------
    x : ndarray
        Real, symmetric matrix
    
    Returns
    -------   
    invsqrt : ndarray
        Input to the power -1/2
    """
    vals, vecs = eigh(x)
    return vecs @ diag(1 / sqrt(vals)) @ vecs.T
