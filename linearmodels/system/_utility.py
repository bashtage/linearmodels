from __future__ import annotations

from typing import cast

import numpy
import numpy as np
from numpy.linalg import inv, matrix_rank
import pandas
import pandas as pd

import linearmodels.typing.data


def blocked_column_product(
    x: linearmodels.typing.data.ArraySequence, s: linearmodels.typing.data.Float64Array
) -> linearmodels.typing.data.Float64Array:
    """
    Parameters
    ----------
    x : list of ndarray
        k-element list of arrays to construct the inner product
    s : ndarray
        Weighting matrix (k by k)

    Returns
    -------
    ndarray
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


def blocked_diag_product(
    x: linearmodels.typing.data.ArraySequence, s: linearmodels.typing.data.Float64Array
) -> linearmodels.typing.data.Float64Array:
    """
    Parameters
    ----------
    x : list of ndarray
        k-element list of arrays to construct the inner product
    s : ndarray
        Weighting matrix (k by k)

    Returns
    -------
    ndarray
        Blocked product.  k x nobs rows and the number of columns is the same
        as the total number of columns in x.
    """
    k = len(x)
    out = []
    for i in range(k):
        row = []
        for j in range(k):
            row.append(s[i, j] * x[j])
        row_arr = np.hstack(row)
        out.append(row_arr)

    return np.vstack(out)


def blocked_inner_prod(
    x: linearmodels.typing.data.ArraySequence, s: linearmodels.typing.data.Float64Array
) -> linearmodels.typing.data.Float64Array:
    r"""
    Parameters
    ----------
    x : list of ndarray
        k-element list of arrays to construct the inner product
    s : ndarray
        Weighting matrix (k by k)

    Returns
    -------
    ndarray
        Weighted inner product constructed from x and s

    Notes
    -----
    Memory efficient implementation of high-dimensional inner product

    .. math::

      X'(S \otimes I_n)X

    where n is the number of observations in the sample
    """
    k = len(x)
    widths = [m.shape[1] for m in x]
    s_is_diag = np.all(np.asarray((s - np.diag(np.diag(s))) == 0.0))

    w0 = widths[0]
    homogeneous = all([w == w0 for w in widths])
    if homogeneous and not s_is_diag:
        # Fast path when all x have same number of columns
        # Slower than diag case when k is large since many 0s
        xa = np.hstack(x)
        return xa.T @ xa * np.kron(s, np.ones((w0, w0)))

    cum_width = np.cumsum([0] + widths)
    total = sum(widths)
    out: np.ndarray = np.zeros((total, total))

    for i in range(k):
        xi = x[i]
        sel_i = slice(cum_width[i], cum_width[i + 1])
        s_ii = s[i, i]
        prod = s_ii * (xi.T @ xi)
        out[sel_i, sel_i] = prod

    # Short circuit if identity
    if s_is_diag:
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
            out[sel_j, sel_i] = prod.T

    return cast(np.ndarray, out)


def blocked_cross_prod(
    x: linearmodels.typing.data.ArraySequence,
    z: linearmodels.typing.data.ArraySequence,
    s: linearmodels.typing.data.Float64Array,
) -> linearmodels.typing.data.Float64Array:
    r"""
    Parameters
    ----------
    x : list of ndarray
        k-element list of arrays to use as the left side of the cross-product
    z : list of ndarray
        k-element list of arrays to use as the right side of the cross-product
    s : ndarray
        Weighting matrix (k by k)

    Returns
    -------
    ndarray
        Weighted cross product constructed from x and s

    Notes
    -----
    Memory efficient implementation of high-dimensional cross product

    .. math::

      X'(S \otimes I_N)Z

    where n is the number of observations in the sample
    """
    k = len(x)
    xp = []
    for i in range(k):
        row = []
        for j in range(k):
            s_ij = s[i, j]
            row.append(s_ij * (x[i].T @ z[j]))
        xp.append(np.concatenate(row, 1))
    return np.concatenate(xp, 0)


def blocked_full_inner_product(
    x: linearmodels.typing.data.Float64Array, s: linearmodels.typing.data.Float64Array
) -> linearmodels.typing.data.Float64Array:
    r"""
    Parameters
    ----------
        x : ndarray
            Array of shape KT by KT
        s : ndarray
            Array of shape K by K

    Notes
    -----
    Computes the quantity

    .. math ::

        x^\prime (S \otimes I_N)x
    """
    k = s.shape[0]
    t = x.shape[0] // k
    sx = np.empty_like(x)
    for i in range(k):
        v = s[i, 0] * x[0:t]
        for j in range(1, k):
            v += s[i, j] * x[j * t : (j + 1) * t]
        sx[i * t : (i + 1) * t] = v
    return x.T @ sx


def inv_matrix_sqrt(
    s: linearmodels.typing.data.Float64Array,
) -> linearmodels.typing.data.Float64Array:
    vecs, vals = np.linalg.eigh(s)
    vecs = 1.0 / np.sqrt(vecs)
    out = vals @ np.diag(vecs) @ vals.T
    return (out + out.T) / 2


class LinearConstraint:
    r"""
    Linear constraint for regression estimation

    Parameters
    ----------
    r : {ndarray, DataFrame}
        Restriction loading matrix
    q : {ndarray, Series}
        Restriction value
    num_params : int
        Number of model parameter.  Used to test for correctness
    require_pandas : bool
        Flag indicating whether r and q must be pandas

    Notes
    -----
    Used to impose the constraints

    .. math ::

        r \beta = q
    """

    def __init__(
        self,
        r: pandas.DataFrame | numpy.ndarray,
        q: pandas.Series | numpy.ndarray | None = None,
        num_params: int | None = None,
        require_pandas: bool = True,
    ) -> None:
        if not isinstance(r, (pd.DataFrame, np.ndarray)):
            raise TypeError("r must be an array or DataFrame")
        elif require_pandas and not isinstance(r, pd.DataFrame):
            raise TypeError("r must be a DataFrame")
        if r.ndim != 2:
            raise ValueError("r must be 2-dimensional")
        r_pd = pd.DataFrame(r)
        ra = np.asarray(r, dtype=np.float64)
        self._r_pd = r_pd
        self._ra = ra
        if q is not None:
            if require_pandas and not isinstance(q, pd.Series):
                raise TypeError("q must be a Series")
            elif not isinstance(q, (pd.Series, np.ndarray)):
                raise TypeError("q must be a Series or an array")
            if r.shape[0] != q.shape[0]:
                raise ValueError("Constraint inputs are not shape compatible")
            q_pd = pd.Series(q, index=r_pd.index)
        else:
            q_pd = pd.Series(np.zeros(r_pd.shape[0]), index=r_pd.index)
        self._q_pd = q_pd
        self._qa = np.asarray(q_pd)
        self._computed = False
        self._t = np.empty((0, 0))
        self._l = np.empty((0, 0))
        self._a = np.empty((0, 0))
        self._num_params = num_params
        self._verify_constraints()

    def __repr__(self) -> str:
        return self.__str__() + "\nid: " + str(hex(id(self)))

    def __str__(self) -> str:
        return f"Linear Constraint with {self._ra.shape[0]} constraints"

    def _verify_constraints(self) -> None:
        r = self._ra
        q = self._qa
        if self._num_params is not None:
            if r.shape[1] != self._num_params:
                raise ValueError(
                    "r is incompatible with the number of model " "parameters"
                )
        rq = np.c_[r, q[:, None]]
        if not np.all(np.isfinite(rq)) or matrix_rank(rq) < rq.shape[0]:
            raise ValueError("Constraints must be non-redundant")
        qr = np.linalg.qr(rq)
        if matrix_rank(qr[1][:, :-1]) != matrix_rank(qr[1]):
            raise ValueError("One or more constraints are infeasible")

    def _compute_transform(self) -> None:
        r = self._ra
        c, k = r.shape
        m = np.eye(k) - r.T @ inv(r @ r.T) @ r
        vals, vecs = np.linalg.eigh(m)
        vals = np.real(vals)
        vecs = np.real(vecs)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        t, left = vecs[:, : k - c], vecs[:, k - c :]
        q = self._qa[:, None]
        a = q.T @ inv(left.T @ r.T) @ left.T
        self._t, self._l, self._a = t, left, a
        self._computed = True

    @property
    def r(self) -> pandas.DataFrame:
        """Constraint loading matrix"""
        return self._r_pd

    @property
    def t(self) -> linearmodels.typing.data.Float64Array:
        """
        Constraint transformation matrix

        Returns
        -------
        ndarray
            Constraint transformation matrix

        Notes
        -----
        Constrained regressors are constructed as x @ t
        """
        if not self._computed:
            self._compute_transform()
        assert isinstance(self._t, np.ndarray)
        return self._t

    @property
    def a(self) -> linearmodels.typing.data.Float64Array:
        r"""
        Transformed constraint target

        Returns
        -------
        ndarray
            Transformed target

        Notes
        -----
        Has two uses.  The first is to translate the restricted parameters
        back to the full parameter vector using

        .. math ::

           beta = t  beta_c + ^\prime

        Also used in estimation or restricted model to demean the dependent
        variable

        .. math ::

            \tilde{y} = y - x  a^\prime
        """
        if not self._computed:
            self._compute_transform()
        assert isinstance(self._a, np.ndarray)
        return self._a

    @property
    def q(self) -> pandas.Series | numpy.ndarray:
        """Constrain target values"""
        return self._q_pd
