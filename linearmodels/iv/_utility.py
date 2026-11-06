from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from formulaic import model_matrix
from formulaic.formula import Formula
from formulaic.materializers.types import NAAction as fNAAction
from formulaic.utils.context import capture_context
import numpy as np
from pandas import DataFrame
from scipy.linalg import qr as sp_qr, solve_triangular

import linearmodels.typing.data

PARSING_ERROR = """
Conversion of formula blocks to DataFrames failed.
The formula blocks used for conversion were:

dependent: {0}
exogenous: {1}
endogenous: {2}
instruments: {3}

The original error was:
"""


SOLVERS = ("legacy", "qr")


def proj(
    y: linearmodels.typing.data.Float64Array,
    x: linearmodels.typing.data.Float64Array,
    *,
    solver: str = "legacy",
) -> linearmodels.typing.data.Float64Array:
    """
    Projection of y on x from y.

    Parameters
    ----------
    y : ndarray
        Array to project (nobs by nseries).
    x : ndarray
        Array to project onto (nobs by nvar).
    solver : {"legacy", "qr"}, optional
        The numerical method. The default is "legacy". See Notes.

    Returns
    -------
    ndarray
        Projected values of y (nobs by nseries).

    Notes
    -----
    "legacy" projects with ``pinv(x)``, which drops the singular values of x that
    are smaller than the largest one times max(nobs, nvar) times the machine
    precision, so that directions are lost if the columns of x are on very
    different scales. "qr" projects on the left singular vectors of x with columns
    scaled to unit length, with the same cutoff, so that the result does not depend
    on the units of the columns of x. Both are the projection on the column space
    of x if x is rank deficient.
    """
    if x.shape[1] == 0:
        return np.zeros_like(y)
    if solver == "legacy":
        return x @ (np.linalg.pinv(x) @ y)
    scale = np.linalg.norm(x, axis=0)
    scale[scale == 0] = 1.0
    u, sv, _ = np.linalg.svd(x / scale, full_matrices=False)
    u = u[:, sv > sv[0] * max(x.shape) * np.finfo(float).eps]
    return u @ (u.T @ y)


def annihilate(
    y: linearmodels.typing.data.Float64Array,
    x: linearmodels.typing.data.Float64Array,
    *,
    solver: str = "legacy",
) -> linearmodels.typing.data.Float64Array:
    """
    Remove projection of y on x from y.

    Parameters
    ----------
    y : ndarray
        Array to project (nobs by nseries).
    x : ndarray
        Array to project onto (nobs by nvar).
    solver : {"legacy", "qr"}, optional
        The numerical method of the projection, see :func:`proj`. The default is
        "legacy".

    Returns
    -------
    ndarray
        Residuals values of y minus y projected on x (nobs by nseries).
    """
    return y - proj(y, x, solver=solver)


def full_column_rank(x: linearmodels.typing.data.Float64Array, tol: float) -> bool:
    """
    Test whether a matrix has full column rank after scaling its columns.

    Parameters
    ----------
    x : ndarray
        Matrix to test (nobs by nvar).
    tol : float
        Relative tolerance. x is rank deficient if the smallest singular value of x
        with columns scaled to unit length is no larger than tol times the largest,
        that is, if the condition number of that matrix is at least 1 / tol.

    Returns
    -------
    bool
        True if x has full column rank.

    Notes
    -----
    The test does not depend on the units of the columns of x. A column that is 0
    makes x rank deficient, and so do fewer rows than columns.
    """
    nobs, nvar = x.shape
    if nobs < nvar:
        return False
    scale = np.linalg.norm(x, axis=0)
    if not np.all(scale > 0):
        return False
    sv = np.linalg.svd(x / scale, compute_uv=False)
    return bool(sv[-1] > tol * sv[0])


def orthonormal_basis(
    x: linearmodels.typing.data.Float64Array,
) -> linearmodels.typing.data.Float64Array:
    """
    Orthonormal basis of the column space of a matrix with full column rank.

    Parameters
    ----------
    x : ndarray
        Matrix with full column rank (nobs by nvar), with nobs >= nvar.

    Returns
    -------
    ndarray
        The nobs by nvar factor Q of the Householder QR decomposition of x.

    Notes
    -----
    Householder QR does not depend on the scale of the columns of x, and the
    rounding error of a projection :math:`QQ'y` grows with the condition number of
    x, not with its square. A projection with ``pinv`` drops the singular values
    that are smaller than the largest one times max(nobs, nvar) times the machine
    precision, so that it loses directions if the columns of x are on very
    different scales.
    """
    return sp_qr(x, mode="economic", check_finite=False)[0]


class KClassFactors(NamedTuple):
    r"""
    Factors of the cross products of the k-class estimator.

    With :math:`M_z = I - QQ'`, where Q is an orthonormal basis of the
    instruments, the factors satisfy

    .. math::

        X'(I - \kappa M_z)X & = R'GR \\
        X'(I - \kappa M_z)y & = R't \\
        QQ'XR^{-1} & = QU

    so that the k-class estimate is :math:`R^{-1}G^{-1}t`.
    """

    r: linearmodels.typing.data.Float64Array
    """Upper triangular factor (nvar by nvar)."""
    g: linearmodels.typing.data.Float64Array | None
    """Symmetric middle factor (nvar by nvar), or None if it is the identity."""
    t: linearmodels.typing.data.Float64Array | None
    """Transformed cross product with y (nvar by 1), or None without y."""
    u: linearmodels.typing.data.Float64Array
    """Coordinates of the projected regressors (ninstr by nvar)."""


def kclass_factors(
    x: linearmodels.typing.data.Float64Array,
    z_basis: linearmodels.typing.data.Float64Array,
    kappa: float,
    y: linearmodels.typing.data.Float64Array | None = None,
) -> KClassFactors:
    r"""
    Factor the cross products of the k-class estimator without forming them.

    Parameters
    ----------
    x : ndarray
        Regressors (nobs by nvar).
    z_basis : ndarray
        Orthonormal basis of the instruments (nobs by ninstr), with ninstr >= nvar.
        See :func:`orthonormal_basis`.
    kappa : float
        The parameter of the k-class estimator.
    y : ndarray, optional
        The dependent variable (nobs by 1). t is only computed if it is provided.

    Returns
    -------
    KClassFactors
        The factors R, G, t and U.

    Notes
    -----
    Let :math:`A = Q'X = Q_A R_A` and :math:`a = Q'y`.

    * :math:`\kappa = 1` (2SLS): :math:`R = R_A`, G is the identity, :math:`t =
      Q_A'a` and :math:`U = Q_A`. The estimate is the least squares solution of
      the projected problem, as in a 2SLS by QR.
    * :math:`\kappa < 1`: the cross products are :math:`(1-\kappa)X'X +
      \kappa A'A`, which are those of the least squares problem with the stacked
      regressors :math:`[\sqrt{1-\kappa}R_X; \sqrt{\kappa}A]` and dependent
      variable :math:`[\sqrt{1-\kappa}Q_X'y; \sqrt{\kappa}a]`, where :math:`X =
      Q_XR_X`. R is the triangular factor of the stacked regressors, which is
      :math:`R_X` if :math:`\kappa = 0` (OLS), and G is the identity.
    * :math:`\kappa > 1` (LIML): :math:`R = R_A`, :math:`F = M_zXR^{-1}`,
      :math:`G = I - (\kappa - 1)F'F`, :math:`t = Q_A'a - (\kappa - 1)F'M_zy` and
      :math:`U = Q_A`.

    R is only used in triangular solves, so that the condition number of the
    problem is not squared, and the factors do not depend on the units of the
    columns of X, y or the instruments.
    """
    nvar = x.shape[1]
    if nvar == 0:
        t0 = None if y is None else np.empty((0,) + y.shape[1:])
        return KClassFactors(
            np.empty((0, 0)), None, t0, np.empty((z_basis.shape[1], 0))
        )
    qx = z_basis.T @ x
    if kappa < 1:
        q_x, r_x = sp_qr(x, mode="economic", check_finite=False)
        t = None
        if kappa == 0:
            r = r_x
            if y is not None:
                t = q_x.T @ y
        else:
            w0, w1 = float(np.sqrt(1 - kappa)), float(np.sqrt(kappa))
            q_s, r = sp_qr(np.vstack([w0 * r_x, w1 * qx]), mode="economic")
            if y is not None:
                t = q_s.T @ np.vstack([w0 * (q_x.T @ y), w1 * (z_basis.T @ y)])
        u = solve_triangular(r, qx.T, trans="T").T
        return KClassFactors(r, None, t, u)

    q_a, r = sp_qr(qx, mode="economic")
    t = None if y is None else q_a.T @ (z_basis.T @ y)
    g = None
    if kappa != 1:
        resid_x = x - z_basis @ qx
        f = solve_triangular(r, resid_x.T, trans="T").T
        g = np.eye(nvar) - (kappa - 1) * (f.T @ f)
        if t is not None and y is not None:
            resid_y = y - z_basis @ (z_basis.T @ y)
            t = t - (kappa - 1) * (f.T @ resid_y)
    return KClassFactors(r, g, t, q_a)


def kclass_solve(factors: KClassFactors) -> linearmodels.typing.data.Float64Array:
    """
    The k-class estimate from the factors of its cross products.

    Parameters
    ----------
    factors : KClassFactors
        Factors computed with y, see :func:`kclass_factors`.

    Returns
    -------
    ndarray
        The estimate :math:`R^{-1}G^{-1}t` (nvar by 1).
    """
    r, g, t = factors.r, factors.g, factors.t
    if t is None:
        raise ValueError("The factors must be computed with y.")
    if r.shape[0] == 0:
        return t
    if g is not None:
        t = np.linalg.solve(g, t)
    return solve_triangular(r, t)


class IVFormulaParser:
    """
    Parse formulas for OLS and IV models.

    Parameters
    ----------
    formula : str
        String formula object.
    data : DataFrame
        Frame containing values for variables used in formula.
    eval_env : int, optional
        Stack depth to use when evaluating formulas. The default is 2.
    context : mapping of str to object, optional
        Variables to use when evaluating the formula, in addition to ``data``. If
        not provided, the variables of the stack frame that ``eval_env`` selects
        are used.

    Raises
    ------
    ValueError
        If the formula does not have the form of an IV formula.

    Notes
    -----
    The general structure of a formula is `dep ~ exog + [endog ~ instr]`.
    """

    def __init__(
        self,
        formula: str,
        data: DataFrame,
        eval_env: int = 2,
        context: Mapping[str, Any] | None = None,
    ):
        self._formula = formula
        self._data = data
        self._eval_env = eval_env
        if not context:
            self._context = capture_context(context=self._eval_env)
        else:
            self._context = context
        self._components: dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        blocks = self._formula.strip().split("~")
        if len(blocks) == 2:
            dep = blocks[0].strip()
            exog = blocks[1].strip()
            endog = "0"
            instr = "0"
        elif len(blocks) == 3:
            blocks = [bl.strip() for bl in blocks]
            if "[" not in blocks[1] or "]" not in blocks[2]:
                raise ValueError(
                    "formula not understood. Endogenous variables and "
                    "instruments must be segregated in a block that "
                    "starts with [ and ends with ]."
                )
            dep = blocks[0].strip()
            exog, endog = (bl.strip() for bl in blocks[1].split("["))
            instr, exog2 = (bl.strip() for bl in blocks[2].split("]"))
            if endog[0] == "+" or endog[-1] == "+":
                raise ValueError(
                    "endogenous block must not start or end with +. This block "
                    "was: {}".format(endog)
                )
            if instr[0] == "+" or instr[-1] == "+":
                raise ValueError(
                    "instrument block must not start or end with +. This "
                    "block was: {}".format(instr)
                )
            if exog:
                exog = exog[:-1].strip() if exog[-1] == "+" else exog
            if exog2:
                exog += exog2
            exog = "0" if not exog else "0 + " + exog
        else:
            raise ValueError("formula contains more than 2 separators (~)")
        comp = {
            "dependent": "0 + " + dep,
            "exog": exog,
            "endog": endog,
            "instruments": instr,
        }
        self._components = comp

    @property
    def eval_env(self) -> int:
        """Set or get the eval env depth."""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value: int) -> None:
        self._eval_env = value

    @property
    def data(
        self,
    ) -> tuple[DataFrame, DataFrame | None, DataFrame | None, DataFrame | None]:
        """Return a tuple containing the dependent, exog, endog and instruments."""
        self._eval_env += 1
        out = self.dependent, self.exog, self.endog, self.instruments
        self._eval_env -= 1
        return out

    @property
    def dependent(self) -> DataFrame:
        """Dependent variable."""
        dep_fmla = self.components["dependent"]
        dep = model_matrix(
            dep_fmla,
            self._data,
            context=self._context,
            ensure_full_rank=False,
            na_action=fNAAction("ignore"),
        )
        return DataFrame(dep)

    @property
    def exog(self) -> DataFrame | None:
        """Exogenous variables."""
        exog_fmla = self.components["exog"]
        exog = Formula(exog_fmla).get_model_matrix(
            self._data,
            context=self._context,
            ensure_full_rank=True,
            na_action=fNAAction("ignore"),
        )
        return self._empty_check(DataFrame(exog))

    @property
    def endog(self) -> DataFrame | None:
        """Endogenous variables."""
        endog_fmla = "0 +" + self.components["endog"]
        endog = Formula(endog_fmla).get_model_matrix(
            self._data,
            context=self._context,
            ensure_full_rank=False,
            na_action=fNAAction("ignore"),
        )
        return self._empty_check(DataFrame(endog))

    @property
    def instruments(self) -> DataFrame | None:
        """Instruments."""
        instr_fmla = "0 +" + self.components["instruments"]
        instr = Formula(instr_fmla).get_model_matrix(
            self._data,
            context=self._context,
            ensure_full_rank=False,
            na_action=fNAAction("ignore"),
        )
        return self._empty_check(DataFrame(instr))

    @property
    def components(self) -> dict[str, str]:
        """Dictionary containing the string components of the formula."""
        return self._components

    @staticmethod
    def _empty_check(arr: DataFrame) -> DataFrame | None:
        return None if arr.shape[1] == 0 else arr
