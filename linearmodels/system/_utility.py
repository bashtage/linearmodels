import numpy as np
import pandas as pd
from numpy import cumsum, diag, eye, zeros
from numpy.linalg import inv
from linearmodels.utility import matrix_rank

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


class LinearConstraint(object):
    # TODO: Docs
    # TODO: Test
    def __init__(self, r, q=None, num_params=None, require_pandas=True):
        if not isinstance(r, (pd.DataFrame, np.ndarray)):
            raise ValueError('r must be an array or DataFrame')
        elif require_pandas and not isinstance(r, pd.DataFrame):
            raise ValueError('r must be a DataFrame')
        r_pd = pd.DataFrame(r)
        ra = np.asarray(r, dtype=np.float64)
        self._r_pd = r_pd
        self._ra = ra
        if q is not None:
            if require_pandas and not isinstance(q, pd.Series):
                raise ValueError('q must be a Series')
            elif not isinstance(q, (pd.Series, np.ndarray)):
                raise ValueError('q must be a Series')
            q_pd = pd.Series(q, index=r_pd.index)
        else:
            q_pd = pd.Series(np.zeros(r_pd.shape[0]), index=r_pd.index)
        self._q_pd = q_pd
        self._qa = np.asarray(q)
        self._t = self._l = self._a = None
        self._num_params = num_params
    
    def _verify_constraints(self):
        r = self._ra
        q = self._qa
        if r.shape[0] != q.shape[0]:
            raise ValueError('Constraint inputs are not shape compatible')
        if self._num_params is not None
            if r.shape[1] != self._num_params:
                raise ValueError('r is incompatible with the number of model '
                                 'parameters')
        rq = np.c_[r, q[:, None]]
        if matrix_rank(rq) < rq.shape[0]:
            raise ValueError('Constraints must be non-redundant')
        qr = np.linalg.qr(rq)
        if matrix_rank(qr[1][:, :-1]) != matrix_rank(qr[1]):
            raise ValueError('One or more constraints are infeasible')
    
    def _compute_transform(self):
        r = self._ra
        c, k = r.shape
        m = eye(k) - r.T @ inv(r @ r.T) @ r
        vals, vecs = np.linalg.eig(m)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:,idx]
        t, l = vecs[:, :k-c], vecs[:,k-c:]
        q = self._qa[:,None]
        a = q.T @ inv(l.T @ r.T) @ l.T
        self._t, self._l, self._a = t, l, a
    
    @property
    def r(self):
        return self._r_pd

    @property
    def t(self):
        if self._t is None:
            self._compute_transform()
        return self._t

    @property
    def a(self):
        if self._a is None:
            self._compute_transform()
        return self._a
    
if __name__ == '__main__':
    import numpy as np
    
    r = np.zeros((2, 10))
    r[0, 0] = 1
    r[1, 1] = 1
    q = np.zeros(2)
    q[1] = 3
    lc = LinearConstraint(r, q, require_pandas=False)
    
    print(lc.a)
    
    x=np.random.randn(1000,10)
    y = np.random.randn(1000,1)
    b = np.linalg.lstsq(x,y)[0]
    m = np.hstack([lc.t, r.T])
    print(b.T @ m)
    bc = np.linalg.lstsq(x@lc.t,y)[0]
    b = (bc.T @ lc.t.T + lc.a).T
    print(bc)
    print(b)
    
    from linearmodels.iv.model import _OLS
    res = _OLS(y,x@lc.t).fit(cov_type='robust')
    
    
    print(res.params.values[:,None].T @ lc.t.T + lc.a).T
    
    xpx = (x @ lc.t).T @ (x @ lc.t)
    xpy = (x @ lc.t).T @ (y - x@lc.a.T)
    
    bc = np.linalg.solve(xpx,xpy)
    print(lc.t @ bc + lc.a.T)
