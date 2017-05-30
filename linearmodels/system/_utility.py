import numpy as np
from numpy import cumsum, diag, zeros


def blocked_column_product(x, s):
    """
    Parameters
    ----------
    x : list of ndarray
        k-element list of arrays to construct the inner product
    s : ndarray
        Weighting matrix (k by k)

    Returns
    -------
    bp : ndarray
        Blocked product.  k x nobs rows and the number of columns is the same
        the number of columns as any member of x.
    """
    k = len(x)
    out = []
    for i in range(k):
        val = s[i, 0] * x[0]
        for j in range(1, k):
            val += s[i, j] * x[j]
        out.append(val)
    return np.vstack(out)


def blocked_diag_product(x, s):
    """
    Parameters
    ----------
    x : list of ndarray
        k-element list of arrays to construct the inner product
    s : ndarray
        Weighting matrix (k by k)

    Returns
    -------
    bp : ndarray
        Blocked product.  k x nobs rows and the number of columns is the same
        as the total number of columns in x.
    """
    k = len(x)
    out = []
    for i in range(k):
        row = []
        for j in range(k):
            row.append(s[i, j] * x[j])
        row = np.hstack(row)
        out.append(row)
    out = np.vstack(out)

    return out


def blocked_inner_prod(x, s):
    """
    Parameters
    ----------
    x : list of ndarray
        k-element list of arrays to construct the inner product
    s : ndarray
        Weighting matrix (k by k)

    Returns
    -------
    ip : ndarray
        Weighted inner product constructed from x and s

    Notes
    -----
    Memory efficient implementation of high-dimensional inner product

    .. math::

      X'(S \otimes I_n)X

    where n is the number of observations in the sample
    """
    k = len(x)
    widths = list(map(lambda m: m.shape[1], x))
    cum_width = cumsum([0] + widths)
    total = sum(widths)
    out = zeros((total, total))

    for i in range(k):
        xi = x[i]
        sel_i = slice(cum_width[i], cum_width[i + 1])
        s_ii = s[i, i]
        prod = s_ii * (xi.T @ xi)
        out[sel_i, sel_i] = prod

    # Short circuit if identity
    if np.all((s - diag(diag(s))) == 0.0):
        return out

    for i in range(k):
        xi = x[i]
        sel_i = slice(cum_width[i], cum_width[i + 1])
        for j in range(i + 1, k):
            sel_j = slice(cum_width[j], cum_width[j + 1])
            xj = x[j]
            s_ij = s[i, j]
            prod = s_ij * (xi.T @ xj)
            out[sel_i, sel_j] = prod
            if i != j:
                out[sel_j, sel_i] = prod.T

    return out


def inv_matrix_sqrt(s):
    vecs, vals = np.linalg.eigh(s)
    vecs = 1.0 / np.sqrt(vecs)
    out = vals @ diag(vecs) @ vals.T
    return (out + out.T) / 2
