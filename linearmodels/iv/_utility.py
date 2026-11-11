from __future__ import annotations

from collections.abc import Mapping
import os
import sys
from types import FrameType
from typing import Any, Literal, NamedTuple, cast
import warnings

from formulaic import model_matrix
from formulaic.formula import Formula
from formulaic.materializers.types import NAAction as fNAAction
from formulaic.utils.context import capture_context
import numpy as np
from pandas import DataFrame
from scipy.linalg import qr as sp_qr, qr_multiply, solve_triangular

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


METHODS = ("pinv", "qr")

FUTURE_DEFAULT_WARNING = (
    'The default method of {name} will change from "pinv" to "qr" in linearmodels '
    '8.0. Pass method="qr" to use the new method now, or method="pinv" to keep the '
    "current one and silence this warning."
)

_IV_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def _external_stacklevel() -> int:
    """The stacklevel, seen from the caller, of the first frame outside iv/."""
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while (
        frame is not None
        and os.path.dirname(os.path.abspath(frame.f_code.co_filename)) == _IV_DIRECTORY
    ):
        frame = frame.f_back
        level += 1
    return level


def resolve_method(
    method: Literal["pinv", "qr"] | None, name: str
) -> Literal["pinv", "qr"]:
    """
    Validate the method keyword, and warn if it is not given.

    Parameters
    ----------
    method : {"pinv", "qr"}, optional
        The numerical method, not case sensitive.
    name : str
        The name of the class or function, used in the warning.

    Returns
    -------
    {"pinv", "qr"}
        The method in lower case, "pinv" if method is None.

    Raises
    ------
    ValueError
        If method is not "pinv", "qr" or None.

    Warns
    -----
    FutureWarning
        If method is None, since the default becomes "qr" in linearmodels 8.0. Code
        inside linearmodels always passes method.
    """
    if method is None:
        message = FUTURE_DEFAULT_WARNING.format(name=name)
        warnings.warn(message, FutureWarning, stacklevel=_external_stacklevel())
        return "pinv"
    if not isinstance(method, str) or method.lower() not in METHODS:
        raise ValueError(f'method must be "pinv", "qr" or None, but it is {method!r}.')
    return cast('Literal["pinv", "qr"]', method.lower())


def proj(
    y: linearmodels.typing.data.Float64Array,
    x: linearmodels.typing.data.Float64Array,
    *,
    method: Literal["pinv", "qr"] = "pinv",
) -> linearmodels.typing.data.Float64Array:
    """
    Projection of y on x from y.

    Parameters
    ----------
    y : ndarray
        Array to project (nobs by nseries).
    x : ndarray
        Array to project onto (nobs by nvar).
    method : {"pinv", "qr"}, optional
        The numerical method. The default is "pinv". See Notes.

    Returns
    -------
    ndarray
        Projected values of y (nobs by nseries).

    Notes
    -----
    Both methods drop the directions of x whose singular values are below
    max(nobs, nvar) * eps times the largest. "pinv" uses the columns of x as they
    are, so it can drop a column that is on a much smaller scale than the others.
    "qr" scales the columns to unit length first, so it does not depend on their
    units.
    """
    if x.shape[1] == 0:
        return np.zeros_like(y)
    if method == "pinv":
        return x @ (np.linalg.pinv(x) @ y)
    scale = np.linalg.norm(x, axis=0)
    scale[scale == 0] = 1.0
    q, r = sp_qr(x / scale, mode="economic", overwrite_a=True, check_finite=False)
    u, sv, _ = np.linalg.svd(r)
    keep = sv > sv[0] * max(x.shape) * np.finfo(float).eps
    if not keep.all():
        # Rank deficient, e.g., a constant and a full set of dummies. x / scale = QR,
        # so Q @ u are its left singular vectors
        q = q @ u[:, keep]
    return q @ (q.T @ y)


def annihilate(
    y: linearmodels.typing.data.Float64Array,
    x: linearmodels.typing.data.Float64Array,
    *,
    method: Literal["pinv", "qr"] = "pinv",
) -> linearmodels.typing.data.Float64Array:
    """
    Remove projection of y on x from y.

    Parameters
    ----------
    y : ndarray
        Array to project (nobs by nseries).
    x : ndarray
        Array to project onto (nobs by nvar).
    method : {"pinv", "qr"}, optional
        The numerical method of the projection, see :func:`proj`. The default is
        "pinv".

    Returns
    -------
    ndarray
        Residuals values of y minus y projected on x (nobs by nseries).
    """
    return y - proj(y, x, method=method)


def rank_tests(
    x: linearmodels.typing.data.Float64Array, tol: float, *, basis: bool = False
) -> tuple[bool, bool, linearmodels.typing.data.Float64Array | None]:
    """
    The rank tests of method="pinv" and method="qr" from one QR decomposition.

    Parameters
    ----------
    x : ndarray
        Matrix to test (nobs by nvar).
    tol : float
        Relative tolerance of the test of method="qr". x is rank deficient if the
        smallest singular value of x with columns scaled to unit length is no larger
        than tol times the largest, that is, if the condition number of that matrix
        is at least 1 / tol.
    basis : bool, optional
        Whether to return the orthonormal basis of x. The default is False.

    Returns
    -------
    pinv_full_rank : bool
        Whether x has full column rank as ``numpy.linalg.matrix_rank`` tests it: no
        singular value is at most the largest times max(nobs, nvar) times the machine
        precision.
    qr_full_rank : bool
        Whether x has full column rank after scaling its columns to unit length. The
        test does not depend on the units of the columns of x.
    q : ndarray or None
        The nobs by nvar factor Q of the Householder QR decomposition of x, which is
        :func:`orthonormal_basis`, if ``basis`` is True and x has full column rank
        for method="qr". None otherwise.

    Notes
    -----
    The singular values of the triangular factor R of x = QR are those of x, with the
    same backward error as an SVD of x, and those of R with its columns scaled by the
    norms of the columns of x are those of the scaled x. A column that is 0 makes x
    rank deficient, and so do fewer rows than columns.
    """
    nobs, nvar = x.shape
    if nobs < nvar:
        return False, False, None
    if basis:
        q, r = sp_qr(x, mode="economic", check_finite=False)
    else:
        q, r = None, sp_qr(x, mode="raw", check_finite=False)[1]
    sv = np.linalg.svd(r, compute_uv=False)
    pinv_full_rank = bool(np.all(sv > sv[0] * max(nobs, nvar) * np.finfo(float).eps))
    scale = np.linalg.norm(x, axis=0)
    if not np.all(scale > 0):
        return pinv_full_rank, False, None
    sv = np.linalg.svd(r / scale, compute_uv=False)
    qr_full_rank = bool(sv[-1] > tol * sv[0])
    return pinv_full_rank, qr_full_rank, q if qr_full_rank else None


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
        Regressors (nobs by nvar), with nvar >= 1.
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
    Let :math:`X = Q_XR_X`, :math:`A = Q'X = Q_A R_A` and :math:`a = Q'y`.

    * :math:`0 \le \kappa < 1`, which includes OLS: the cross products are
      :math:`(1-\kappa)X'X + \kappa A'A`, those of the least squares problem with
      regressors :math:`[\sqrt{1-\kappa}R_X; \sqrt{\kappa}A]` and dependent
      variable :math:`[\sqrt{1-\kappa}Q_X'y; \sqrt{\kappa}a]`. R is the triangular
      factor of these regressors, and G is the identity.
    * Otherwise, which includes 2SLS and LIML: :math:`R = R_A`,
      :math:`F = M_zXR^{-1}`, :math:`G = I - (\kappa - 1)F'F`,
      :math:`t = Q_A'a - (\kappa - 1)F'M_zy` and :math:`U = Q_A`. With
      :math:`\kappa = 1` (2SLS), G is the identity, and the estimate is the least
      squares solution of the projected problem, as in R's ``ivreg``.

    R is only used in triangular solves, so the condition number of the problem is
    not squared, and the factors do not depend on the units of the variables.
    """
    if x.shape[1] == 0:
        raise ValueError('method="qr" requires at least one regressor.')
    qx = z_basis.T @ x
    if 0 <= kappa < 1:
        qx_y, r_x = _triangular_factor(x, y)
        w0, w1 = float(np.sqrt(1 - kappa)), float(np.sqrt(kappa))
        q_s, r = sp_qr(np.vstack([w0 * r_x, w1 * qx]), mode="economic")
        t = None
        if y is not None:
            t = q_s.T @ np.concatenate([w0 * qx_y, w1 * (z_basis.T @ y)])
        u = solve_triangular(r, qx.T, trans="T").T
        return KClassFactors(r, None, t, u)

    q_a, r = sp_qr(qx, mode="economic")
    t = None if y is None else q_a.T @ (z_basis.T @ y)
    g = None
    if kappa != 1:
        f = solve_triangular(r, (x - z_basis @ qx).T, trans="T").T
        g = np.eye(x.shape[1]) - (kappa - 1) * (f.T @ f)
        if y is not None and t is not None:
            resid_y = y - z_basis @ (z_basis.T @ y)
            t = t - (kappa - 1) * (f.T @ resid_y)
    return KClassFactors(r, g, t, q_a)


def _triangular_factor(
    x: linearmodels.typing.data.Float64Array,
    y: linearmodels.typing.data.Float64Array | None,
) -> tuple[Any, linearmodels.typing.data.Float64Array]:
    """Q'y and R of the Householder QR decomposition of x, without forming Q."""
    if y is None:
        return None, sp_qr(x, mode="raw", check_finite=False)[1]
    qy, r = qr_multiply(x, y.reshape(y.shape[0], -1).T, mode="right")
    return (qy[0] if y.ndim == 1 else qy.T), r


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
    t = cast("linearmodels.typing.data.Float64Array", factors.t)
    if factors.g is not None:
        t = np.linalg.solve(factors.g, t)
    return solve_triangular(factors.r, t)


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
