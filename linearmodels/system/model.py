"""
Estimators for systems of equations.

References
----------
Greene, William H. "Econometric analysis 4th edition." International edition,
    New Jersey: Prentice Hall (2000).
StataCorp. 2013. Stata 13 Base Reference Manual. College Station, TX: Stata
    Press.
Henningsen, A., & Hamann, J. (2007). systemfit: A Package for Estimating
    Systems of Simultaneous Equations in R. Journal of Statistical Software,
    23(4), 1 - 40. doi:http://dx.doi.org/10.18637/jss.v023.i04
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce
import textwrap
from typing import Any, Literal, Optional, TypedDict, cast
import warnings

from formulaic.utils.context import capture_context
import numpy as np
from numpy.linalg import inv, lstsq, matrix_rank, solve
from pandas import DataFrame, Index, Series, concat

from linearmodels.iv._utility import IVFormulaParser
from linearmodels.iv.covariance import KERNEL_LOOKUP
from linearmodels.iv.data import IVData
from linearmodels.iv.gmm import _uses_default_bandwidth, _warn_default_bandwidth
from linearmodels.shared.alignment import (
    check_alignment,
    check_row_alignment,
    reference_labels,
    row_labels,
)
from linearmodels.shared.exceptions import (
    check_cov_config,
    lower_option,
    missing_warning,
    unknown_option_error,
)
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.shared.inputs import as_array_like, as_float64
from linearmodels.shared.linalg import has_constant
from linearmodels.shared.rsquared import r_squared, total_ss_is_zero
from linearmodels.shared.utility import AttrDict
from linearmodels.shared.weights import check_weights, normalize_weights
from linearmodels.system._utility import (
    LinearConstraint,
    blocked_column_product,
    blocked_cross_prod,
    blocked_diag_product,
    blocked_inner_prod,
    inv_matrix_sqrt,
)
from linearmodels.system.covariance import (
    ClusteredCovariance,
    GMMHeteroskedasticCovariance,
    GMMHomoskedasticCovariance,
    GMMKernelCovariance,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    KernelCovariance,
)
from linearmodels.system.gmm import (
    HeteroskedasticWeightMatrix,
    HomoskedasticWeightMatrix,
    KernelWeightMatrix,
)
from linearmodels.system.results import GMMSystemResults, SystemResults
import linearmodels.typing.data

__all__ = ["IV3SLS", "SUR", "IVSystemGMM", "LinearConstraint"]

UNKNOWN_EQ_TYPE = """\
Contents of each equation must be either a dictionary with keys "dependent"
and "exog" or a 2-element tuple of he form (dependent, exog).
equations[{key}] was {type}
"""

COV_TYPES = {
    "unadjusted": "unadjusted",
    "homoskedastic": "unadjusted",
    "robust": "robust",
    "heteroskedastic": "robust",
    "kernel": "kernel",
    "hac": "kernel",
    "clustered": "clustered",
}

COV_EST = {
    "unadjusted": HomoskedasticCovariance,
    "robust": HeteroskedasticCovariance,
    "kernel": KernelCovariance,
    "clustered": ClusteredCovariance,
}

GMM_W_EST = {
    "unadjusted": HomoskedasticWeightMatrix,
    "robust": HeteroskedasticWeightMatrix,
    "kernel": KernelWeightMatrix,
}

GMM_COV_EST = {
    "unadjusted": GMMHomoskedasticCovariance,
    "robust": GMMHeteroskedasticCovariance,
    "kernel": GMMKernelCovariance,
}


class _WeightConfig(TypedDict, total=False):
    """The options of the unadjusted and robust weight matrices of IVSystemGMM."""

    center: bool
    debiased: bool


class _KernelWeightConfig(_WeightConfig, total=False):
    """The options of the kernel weight matrix of IVSystemGMM."""

    kernel: str
    bandwidth: float | None
    optimal_bw: bool | None


class _GMMCovConfig(TypedDict, total=False):
    """The options of the unadjusted and robust covariances of IVSystemGMM."""

    debiased: bool


class _GMMKernelCovConfig(_GMMCovConfig, total=False):
    """The options of the kernel covariance of IVSystemGMM."""

    kernel: str
    bandwidth: float | None


_LS_KEYS = {
    "unadjusted": ("debiased",),
    "robust": ("debiased",),
    "kernel": ("debiased", "kernel", "bandwidth"),
    "clustered": ("debiased", "clusters", "group_debias"),
}
_LS_COV_KEYS = {name: _LS_KEYS[cov] for name, cov in COV_TYPES.items()}
_GMM_COV_KEYS = {
    name: keys for name, keys in _LS_COV_KEYS.items() if COV_TYPES[name] in GMM_W_EST
}

# The names in COV_TYPES that IVSystemGMM supports, which excludes "clustered"
GMM_TYPES = {name: cov for name, cov in COV_TYPES.items() if cov in GMM_W_EST}


def _missing_weights(
    weights: Mapping[str, linearmodels.typing.data.ArrayLike | None],
) -> None:
    """Raise warning if missing weights found."""
    missing = [key for key in weights if weights[key] is None]
    if missing:
        msg = "Weights not found for equation labels:\n{}".format(", ".join(missing))
        warnings.warn(msg, UserWarning, stacklevel=2)


def _parameters_from_xprod(
    xpx: linearmodels.typing.data.Float64Array,
    xpy: linearmodels.typing.data.Float64Array,
    constraints: LinearConstraint | None = None,
) -> linearmodels.typing.data.Float64Array:
    r"""
    Estimate regression parameters from cross products.

    Parameters
    ----------
    xpx : ndarray
        Cross product measuring variation in x (nvar by nvar).
    xpy : ndarray
        Cross product measuring covariation between x and y (nvar by 1).
    constraints : LinearConstraint, optional
        Constraints to use in estimation. The default is no constraints.

    Returns
    -------
    ndarray
        Estimated parameters (nvar by 1).

    Notes
    -----
    xpx and xpy can be any form similar to the two inputs into the usual
    parameter estimator for a linear regression. In particular, many
    estimators can be written as

    .. math::

        (x^\prime w x)^{-1}(x^\prime w y)

    for some weight matrix :math:`w`.
    """
    if constraints is not None:
        cons = constraints
        xpy = cons.t.T @ xpy - cons.t.T @ xpx @ cons.a.T
        xpx = cons.t.T @ xpx @ cons.t
        params_c = solve(xpx, xpy)
        params = cons.t @ params_c + cons.a.T
    else:
        params = solve(xpx, xpy).astype(float, copy=False)
    return params


class SystemFormulaParser:
    """
    Parser for the formulas of a system of equations.

    Parameters
    ----------
    formula : str or dict of str to str
        Either a string with the equations in curly braces, or a dictionary of
        formulas where each key is an equation label.
    data : DataFrame
        Frame containing named variables.
    weights : dict of str to array_like, optional
        Weights of the equations, where each key is an equation label. The
        default is no weights.
    eval_env : int, optional
        Depth of the frame, relative to the caller of the parser, used to
        evaluate the formulas when ``context`` is not given. The default is 2.
    context : dict of str to object, optional
        Namespace used to evaluate the formulas. The default is the namespace
        of the frame at depth ``eval_env``.
    """

    def __init__(
        self,
        formula: Mapping[str, str] | str,
        data: DataFrame,
        weights: Mapping[str, linearmodels.typing.data.ArrayLike] | None = None,
        eval_env: int = 2,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(formula, (Mapping, str)):
            raise TypeError("formula must be a string or dictionary-like")
        if weights is not None and not hasattr(weights, "keys"):
            raise TypeError(
                "weights must be dictionary-like, such as a dict or a DataFrame, "
                "where the key of each value is an equation label. "
                f"Got {type(weights).__name__}."
            )
        self._formula: Mapping[str, str] | str = formula
        self._data = data
        self._weights = weights
        self._parsers: dict[str, IVFormulaParser] = {}
        self._weight_dict: dict[str, Series | None] = {}
        self._eval_env = eval_env
        self._clean_formula: dict[str, str] = {}
        if context is None:
            self._context = capture_context(eval_env)
        else:
            self._context = context
        self._parse()

    @staticmethod
    def _prevent_autoconst(formula: str) -> str:
        if not (" 0+" in formula or " 0 +" in formula):
            formula = "~ 0 +".join(formula.split("~"))
        return formula

    @staticmethod
    def _convert_to_series(
        value: linearmodels.typing.data.ArrayLike, name: str
    ) -> Series:
        value = as_array_like(value, name)
        if isinstance(value, np.ndarray):
            value = as_float64(value, name)
        shape = value.shape
        if len(shape) > 2 or (len(shape) == 2 and min(shape) != 1):
            raise ValueError(f"{name} must be squeezable to 1D.")
        if len(shape) == 1:
            value_series = Series(value)
        elif isinstance(value, DataFrame):
            value_series = value.iloc[:, 0]
        else:
            value_series = Series(np.squeeze(value))

        return value_series

    def _parse(self) -> None:
        formula = self._formula
        data = self._data
        weights = self._weights
        parsers = self._parsers
        weight_dict = self._weight_dict
        cln_formula = self._clean_formula

        if isinstance(formula, Mapping):
            for formula_key in formula:
                f = formula[formula_key]
                f = self._prevent_autoconst(f)
                parsers[formula_key] = IVFormulaParser(f, data, context=self._context)

                if weights is not None:
                    if formula_key in weights:
                        value = weights[formula_key]
                        weight_dict[formula_key] = self._convert_to_series(
                            value, "weights"
                        )
                    else:
                        weight_dict[formula_key] = None
                cln_formula[formula_key] = f
        else:
            formula = formula.replace("\n", " ").strip()
            parts = formula.split("}")
            for _part in parts:
                key = base_key = None
                part = _part.strip()
                if part == "":
                    continue
                part = part.replace("{", "")
                if ":" in part.split("~")[0]:
                    base_key, part = part.split(":")
                    key = base_key = base_key.strip()
                    part = part.strip()
                f = self._prevent_autoconst(part)
                if base_key is None:
                    base_key = key = f.split("~")[0].strip()
                count = 0
                while key in parsers:
                    key = base_key + f".{count}"
                    count += 1
                assert isinstance(key, str)
                parsers[key] = IVFormulaParser(f, data, context=self._context)
                cln_formula[key] = f
                if weights is not None:
                    if key in weights:
                        value = weights[key]
                        weight_dict[key] = self._convert_to_series(value, "weights")
                    else:
                        weight_dict[key] = None
        _missing_weights(weight_dict)
        self._weight_dict = weight_dict

    def _get_variable(self, variable: str) -> dict[str, DataFrame | None]:
        return {key: getattr(self._parsers[key], variable) for key in self._parsers}

    @property
    def formula(self) -> dict[str, str]:
        """Cleaned version of formula."""
        return self._clean_formula

    @property
    def eval_env(self) -> int:
        """Set or get the eval env depth."""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value: int) -> None:  # numpydoc ignore=GL08
        self._eval_env = value
        self._context = capture_context(self._eval_env)
        # Update parsers for new level
        parsers = self._parsers
        new_parsers = {}
        for key in parsers:
            parser = parsers[key]
            new_parsers[key] = IVFormulaParser(
                parser._formula, parser._data, context=self._context
            )
        self._parsers = new_parsers

    @property
    def equation_labels(self) -> list[str]:
        """Labels of the equations."""
        return list(self._parsers.keys())

    @property
    def data(self) -> dict[str, dict[str, linearmodels.typing.data.ArrayLike | None]]:
        """Dependent, exogenous, endogenous, instruments and weights by equation."""
        out: dict[str, dict[str, linearmodels.typing.data.ArrayLike | None]] = {}
        dep = self.dependent
        for key in dep:
            out[key] = {"dependent": dep[key]}
        exog = self.exog
        for key in exog:
            out[key]["exog"] = exog[key]
        endog = self.endog
        for key in endog:
            out[key]["endog"] = endog[key]
        instr = self.instruments
        for key in instr:
            out[key]["instruments"] = instr[key]
        for key in self._weight_dict:
            if self._weight_dict[key] is not None:
                out[key]["weights"] = self._weight_dict[key]
        return out

    @property
    def dependent(self) -> dict[str, DataFrame | None]:
        """Dependent variable of each equation."""
        return self._get_variable("dependent")

    @property
    def exog(self) -> dict[str, DataFrame | None]:
        """Exogenous regressors of each equation."""
        return self._get_variable("exog")

    @property
    def endog(self) -> dict[str, DataFrame | None]:
        """Endogenous regressors of each equation."""
        return self._get_variable("endog")

    @property
    def instruments(self) -> dict[str, DataFrame | None]:
        """Instruments of each equation."""
        return self._get_variable("instruments")


class _SystemModelBase:
    r"""
    Base class for system estimators

    Parameters
    ----------
    equations : dict of str to tuple or dict
        Dictionary-like structure containing dependent, exogenous, endogenous
        and instrumental variables.  Each key is an equations label and must
        be a string. Each value must be either a tuple of the form (dependent,
        exog, endog, instrument[, weights]) or a dictionary with keys "dependent",
        and at least one of "exog" or "endog" and "instruments".  When using a
        tuple, values must be provided for all 4 variables, although either
        empty arrays or `None` can be passed if a category of variable is not
        included in a model. The dictionary may contain optional keys for
        "exog", "endog", "instruments", and "weights". "exog" can be omitted
        if all variables in an equation are endogenous. Alternatively, "exog"
        can contain either an empty array or `None` to indicate that an
        equation contains no exogenous regressors. Similarly "endog" and
        "instruments" can either be omitted or may contain an empty array (or
        `None`) if all variables in an equation are exogenous. The variables are
        array_like, such as lists, NumPy arrays, pandas objects and xarray
        DataArrays, and a ``TypeError`` that names the variable is raised for
        complex values or text.
    sigma : array_like, optional
        Prespecified residual covariance to use in GLS estimation, with shape
        (nequations, nequations). If not provided, FGLS is implemented based on an
        estimate of sigma.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> None:
        if not isinstance(equations, Mapping):
            raise TypeError("equations must be a dictionary-like")
        for key in equations:
            if not isinstance(key, str):
                raise ValueError("Equation labels (keys) must be strings")

        self._equations = equations
        self._sigma = None
        if sigma is not None:
            self._sigma = np.asarray(sigma)
            k = len(self._equations)
            if self._sigma.shape != (k, k):
                raise ValueError(
                    "sigma must be a square matrix with dimensions "
                    "equal to the number of equations"
                )
        self._param_names: list[str] = []
        self._eq_labels: list[str] = []
        self._dependent: list[IVData] = []
        self._exog: list[IVData] = []
        self._instr: list[IVData] = []
        self._endog: list[IVData] = []

        self._y: list[linearmodels.typing.data.Float64Array] = []
        self._x: list[linearmodels.typing.data.Float64Array] = []
        self._wy: list[linearmodels.typing.data.Float64Array] = []
        self._wx: list[linearmodels.typing.data.Float64Array] = []
        self._w: list[linearmodels.typing.data.Float64Array] = []
        self._z: list[linearmodels.typing.data.Float64Array] = []
        self._wz: list[linearmodels.typing.data.Float64Array] = []

        self._weights: list[IVData] = []
        self._formula: str | dict[str, str] | None = None
        self._constraints: LinearConstraint | None = None
        self._constant_loc: list[int] = []
        self._has_constant: Series = Series(dtype=bool)
        self._common_exog = False
        self._original_index: Index | None = None
        self._model_name = "Three Stage Least Squares (3SLS)"

        self._validate_data()

    @property
    def formula(self) -> str | dict[str, str] | None:
        """Set or get the formula used to construct the model."""
        return self._formula

    @formula.setter
    def formula(
        self, value: str | dict[str, str] | None
    ) -> None:  # numpydoc ignore=GL08
        self._formula = value

    def _validate_data(self) -> None:
        ids = []
        for i, key in enumerate(self._equations):
            self._eq_labels.append(key)
            eq_data = self._equations[key]
            dep_name = "dependent_" + str(i)
            exog_name = "exog_" + str(i)
            endog_name = "endog_" + str(i)
            instr_name = "instr_" + str(i)
            if isinstance(eq_data, (tuple, list)):
                dep = IVData(eq_data[0], var_name=dep_name)
                self._dependent.append(dep)
                current_id: tuple[int, ...] = (id(eq_data[1]),)
                self._exog.append(
                    IVData(eq_data[1], var_name=exog_name, nobs=dep.shape[0])
                )
                endog = IVData(eq_data[2], var_name=endog_name, nobs=dep.shape[0])
                if endog.shape[1] > 0:
                    current_id += (id(eq_data[2]),)
                self._endog.append(endog)

                instr = IVData(eq_data[3], var_name=instr_name, nobs=dep.shape[0])
                if instr.shape[1] > 0:
                    current_id += (id(eq_data[3]),)
                ids.append(current_id)
                self._instr.append(instr)
                if len(eq_data) == 5:
                    weights_name = "weights_" + str(i)
                    self._weights.append(IVData(eq_data[4], var_name=weights_name))
                else:
                    dep_shape = self._dependent[-1].shape
                    self._weights.append(IVData(np.ones(dep_shape)))

            elif isinstance(eq_data, (dict, Mapping)):
                dep = IVData(eq_data["dependent"], var_name=dep_name)
                self._dependent.append(dep)

                exog = eq_data.get("exog", None)
                self._exog.append(IVData(exog, var_name=exog_name, nobs=dep.shape[0]))
                current_id = (id(exog),)

                endog_values = eq_data.get("endog", None)
                endog = IVData(endog_values, var_name=endog_name, nobs=dep.shape[0])
                self._endog.append(endog)
                if "endog" in eq_data:
                    current_id += (id(eq_data["endog"]),)

                instr_values = eq_data.get("instruments", None)
                instr = IVData(instr_values, var_name=instr_name, nobs=dep.shape[0])
                if "instruments" in eq_data:
                    current_id += (id(instr_values),)
                ids.append(current_id)
                self._instr.append(instr)

                if "weights" in eq_data:
                    weights_name = "weights_" + str(i)
                    self._weights.append(
                        IVData(eq_data["weights"], var_name=weights_name)
                    )
                else:
                    self._weights.append(IVData(np.ones(dep.shape)))
            else:
                msg = UNKNOWN_EQ_TYPE.format(key=key, type=type(eq_data))
                raise TypeError(msg)
        self._has_instruments = False
        for instr in self._instr:
            self._has_instruments = self._has_instruments or (instr.shape[1] > 1)

        for i, comps in enumerate(
            zip(
                self._dependent,
                self._exog,
                self._endog,
                self._instr,
                self._weights,
                strict=False,
            )
        ):
            shapes = [a.shape[0] for a in comps]
            if min(shapes) != max(shapes):
                raise ValueError(
                    "Dependent, exogenous, endogenous and "
                    "instruments, and weights, if provided, do "
                    "not have the same number of observations in "
                    "{eq}".format(eq=self._eq_labels[i])
                )

        self._check_alignment()
        self._drop_missing()
        self._common_exog = len(set(ids)) == 1
        if self._common_exog:
            # Common exog requires weights are also equal
            w0 = self._weights[0].ndarray
            for w in self._weights:
                self._common_exog = self._common_exog and bool(np.all(w.ndarray == w0))
        constant = []
        constant_loc = []

        exog_ivd: IVData
        for dep, exog_ivd, endog, instr, w, label in zip(
            self._dependent,
            self._exog,
            self._endog,
            self._instr,
            self._weights,
            self._eq_labels,
            strict=False,
        ):
            y = cast(
                "linearmodels.typing.data.Float64Array",
                dep.ndarray,
            )
            x = np.concatenate([exog_ivd.ndarray, endog.ndarray], 1, dtype=float)
            z = np.concatenate([exog_ivd.ndarray, instr.ndarray], 1, dtype=float)
            w_arr = cast(
                "linearmodels.typing.data.Float64Array",
                w.ndarray,
            )
            check_weights(w_arr, f"weights in equation {label!r}")
            w_arr = normalize_weights(w_arr)
            w_sqrt = np.sqrt(w_arr)
            self._w.append(w_arr)
            self._y.append(y)
            self._x.append(x)
            self._z.append(z)
            self._wy.append(y * w_sqrt)
            self._wx.append(x * w_sqrt)
            self._wz.append(z * w_sqrt)
            cols = list(exog_ivd.cols) + list(endog.cols)
            self._param_names.extend([label + "_" + col for col in cols])
            if y.shape[0] <= x.shape[1]:
                raise ValueError(
                    "Fewer observations than variables in "
                    "equation `{eq}`".format(eq=label)
                )
            if matrix_rank(x) < x.shape[1]:
                raise ValueError(
                    "Equation `{eq}` regressor array is not full rank".format(eq=label)
                )
            if x.shape[1] > z.shape[1]:
                raise ValueError(
                    "Equation `{eq}` has fewer instruments than "
                    "endogenous variables.".format(eq=label)
                )
            if z.shape[1] > z.shape[0]:
                raise ValueError(
                    "Fewer observations than instruments in "
                    "equation `{eq}`".format(eq=label)
                )
            if matrix_rank(z) < z.shape[1]:
                raise ValueError(
                    "Equation {eq} instrument array is not full rank".format(eq=label)
                )

        for rhs in self._x:
            const, const_loc = has_constant(rhs)
            constant.append(const)
            constant_loc.append(const_loc)
        self._has_constant = Series(
            constant, index=[d.cols[0] for d in self._dependent]
        )
        self._constant_loc = constant_loc

    def _check_alignment(self) -> None:
        """Check that inputs with labels use the labels of dependent, or the first"""
        nobs = [dep.shape[0] for dep in self._dependent]
        for i, label in enumerate(self._eq_labels):
            if nobs[i] != nobs[0]:
                raise ValueError(
                    "All equations must have the same number of observations. "
                    f"The dependent variable in equation {label!r} has {nobs[i]} "
                    f"observations, but in equation {self._eq_labels[0]!r} it has "
                    f"{nobs[0]}."
                )
        # The dependent variables of all equations use the same labels
        reference: Index | None = None
        reference_name = ""
        for i, label in enumerate(self._eq_labels):
            dependent_name = f"dependent in equation {label!r}"
            expected = row_labels(self._dependent[i])
            if expected is None:
                continue
            if reference is None:
                reference, reference_name = expected, dependent_name
            else:
                check_row_alignment(dependent_name, expected, reference, reference_name)
        equation_inputs: list[list[tuple[str, object]]] = [
            [
                (f"exog in equation {label!r}", self._exog[i]),
                (f"endog in equation {label!r}", self._endog[i]),
                (f"instruments in equation {label!r}", self._instr[i]),
                (f"weights in equation {label!r}", self._weights[i]),
            ]
            for i, label in enumerate(self._eq_labels)
        ]
        # If no dependent variable has labels, the first input with labels sets them
        self._labels, self._labels_name = reference_labels(
            reference,
            [item for items in equation_inputs for item in items],
            reference=reference_name,
        )
        for i, label in enumerate(self._eq_labels):
            # The inputs of an equation use the labels of its own dependent variable
            # if it has labels, and the labels of the system otherwise
            expected = row_labels(self._dependent[i])
            if expected is None:
                check_alignment(
                    self._labels, equation_inputs[i], reference=self._labels_name
                )
            else:
                check_alignment(
                    expected,
                    equation_inputs[i],
                    reference=f"dependent in equation {label!r}",
                )

    def _check_clusters(self, config: Mapping[str, Any]) -> None:
        """Check that clusters with labels use the labels of the system"""
        labels = None if self._labels is None else self._labels[~self._drop_locs]
        check_alignment(
            labels, {"clusters": config.get("clusters")}, reference=self._labels_name
        )

    def _drop_missing(self) -> None:
        k = len(self._dependent)
        nobs = self._dependent[0].shape[0]
        self._original_index = Index(self._dependent[0].rows)
        missing = np.zeros(nobs, dtype=bool)

        values = [self._dependent, self._exog, self._endog, self._instr, self._weights]
        for i in range(k):
            for value in values:
                nulls = value[i].isnull
                if nulls.any():
                    missing |= np.asarray(nulls)

        missing_warning(missing, stacklevel=4)
        self._drop_locs = missing
        if np.any(missing):
            for i in range(k):
                self._dependent[i].drop(missing)
                self._exog[i].drop(missing)
                self._endog[i].drop(missing)
                self._instr[i].drop(missing)
                self._weights[i].drop(missing)

    def __repr__(self) -> str:
        return self.__str__() + f"\nid: {hex(id(self))}"

    def __str__(self) -> str:
        out = self._model_name + ", "
        out += f"{len(self._y)} Equations:\n"
        eqns = ", ".join(self._equations.keys())
        out += "\n".join(textwrap.wrap(eqns, 70))
        if self._common_exog:
            out += "\nCommon Exogenous Variables"
        return out

    def predict(
        self,
        params: linearmodels.typing.data.ArrayLike,
        *,
        equations: (
            Mapping[str, Mapping[str, linearmodels.typing.data.ArrayLike]] | None
        ) = None,
        data: DataFrame | None = None,
        eval_env: int = 1,
    ) -> DataFrame:
        """
        Predict values for additional data.

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1).
        equations : dict of str to dict, optional
            Dictionary-like structure containing exogenous and endogenous
            variables.  Each key is an equations label and must
            match the labels used to fit the model. Each value must be a
            dictionary with keys "exog" and "endog". If predictions are not
            required for one of more of the model equations, these keys can
            be omitted.
        data : DataFrame, optional
            Values to use when making predictions from a model constructed
            from a formula.
        eval_env : int, optional
            Depth to use when evaluating formulas. The default is 1.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters, with a column for
            each equation that has predictions.

        Raises
        ------
        ValueError
            If neither ``equations`` nor ``data`` is provided, or if ``params`` does
            not have one element for each parameter of the model.

        Notes
        -----
        If `data` is not none, then `equations` must be none.
        Predictions from models constructed using formulas can
        be computed using either `equations`, which will treat these are
        arrays of values corresponding to the formula-process data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.

        When using `exog` and `endog`, the regressor array for a particular
        equation is assembled as
        `[equations[eqn]["exog"], equations[eqn]["endog"]]` where `eqn` is
        an equation label. These must correspond to the columns in the
        estimated model.
        """

        if data is not None:
            assert self.formula is not None
            parser = SystemFormulaParser(
                self.formula, data=data, context=capture_context(eval_env)
            )
            equations_d = parser.data
        else:
            if equations is None:
                raise ValueError("One of equations or data must be provided.")
            assert equations is not None
            equations_d = {k: dict(v) for k, v in equations.items()}
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        nx = int(sum(_x.shape[1] for _x in self._x))
        if params.shape[0] != nx:
            raise ValueError(
                f"Parameters must have {nx} elements; found {params.shape[0]}."
            )
        loc = 0
        out = {}
        for i, label in enumerate(self._eq_labels):
            kx = self._x[i].shape[1]
            if label in equations_d:
                b = params[loc : loc + kx]
                eqn = equations_d[label]
                exog = eqn.get("exog", None)
                endog = eqn.get("endog", None)
                if exog is None and endog is None:
                    loc += kx
                    continue

                if exog is not None:
                    exog_endog = IVData(exog, var_name="exog").pandas
                    if endog is not None:
                        endog_ivd = IVData(endog, var_name="endog")
                        exog_endog = concat([exog_endog, endog_ivd.pandas], axis=1)
                else:
                    exog_endog = IVData(endog, var_name="endog").pandas

                fitted = np.asarray(exog_endog) @ b
                fitted_df = DataFrame(fitted, index=exog_endog.index, columns=[label])
                out[label] = fitted_df
            loc += kx
        out_df = reduce(
            lambda left, right: left.merge(
                right, how="outer", left_index=True, right_index=True
            ),
            [out[key] for key in out],
        )
        return out_df

    def _multivariate_ls_fit(
        self,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        wy, wx, wxhat = self._wy, self._wx, self._wxhat
        k = len(wxhat)

        xpx = blocked_inner_prod(wxhat, np.eye(len(wxhat)))
        _xpy = [wxhat[i].T @ wy[i] for i in range(k)]
        xpy = np.vstack(_xpy)
        beta = _parameters_from_xprod(xpx, xpy, constraints=self.constraints)

        loc = 0
        eps = []
        for i in range(k):
            nb = wx[i].shape[1]
            b = beta[loc : loc + nb]
            eps.append(wy[i] - wx[i] @ b)
            loc += nb
        eps_arr = np.hstack(eps)

        return beta, eps_arr

    def _construct_xhat(self) -> None:
        k = len(self._x)
        self._xhat = []
        self._wxhat = []
        for i in range(k):
            x, z = self._x[i], self._z[i]
            if z.shape == x.shape and np.all(z == x):
                # OLS, no instruments
                self._xhat.append(x)
                self._wxhat.append(self._wx[i])
            else:
                # Project the weighted regressors on the weighted instruments so that
                # the first stage uses the same weights as the second stage
                wx, wz = self._wx[i], self._wz[i]
                delta = lstsq(wz, wx, rcond=None)[0]
                self._xhat.append(z @ delta)
                self._wxhat.append(wz @ delta)

    def _gls_estimate(
        self,
        eps: linearmodels.typing.data.Float64Array,
        nobs: int,
        total_cols: int,
        ci: Sequence[int],
        full_cov: bool,
        debiased: bool,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """Core estimation routine for iterative GLS"""
        wy, wx, wxhat = self._wy, self._wx, self._wxhat

        if self._sigma is None:
            sigma = eps.T @ eps / nobs
            sigma *= self._sigma_scale(debiased)
        else:
            sigma = self._sigma
        est_sigma = sigma

        if not full_cov:
            sigma = np.diag(np.diag(sigma))
        sigma_inv = cast("linearmodels.typing.data.Float64Array", inv(sigma))

        k = len(wy)

        xpx = blocked_inner_prod(wxhat, sigma_inv)
        xpy = np.zeros((total_cols, 1))
        for i in range(k):
            sy = np.zeros((nobs, 1))
            for j in range(k):
                sy += sigma_inv[i, j] * wy[j]
            xpy[ci[i] : ci[i + 1]] = wxhat[i].T @ sy

        beta = _parameters_from_xprod(xpx, xpy, constraints=self.constraints)

        loc = 0
        for j in range(k):
            _wx = wx[j]
            _wy = wy[j]
            kx = _wx.shape[1]
            eps[:, [j]] = _wy - _wx @ beta[loc : loc + kx]
            loc += kx

        return beta, eps, sigma, est_sigma

    def _multivariate_ls_finalize(
        self,
        beta: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        cov_type: str,
        **cov_config: bool,
    ) -> SystemResults:
        k = len(self._wx)

        # Covariance estimation
        cov_estimator = COV_EST[cov_type]
        cov_est = cov_estimator(
            self._wxhat,
            eps,
            sigma,
            sigma,
            gls=False,
            constraints=self._constraints,
            **cov_config,
        )
        cov = cov_est.cov

        # eps are the weighted residuals. The residuals do not use the weights
        loc = 0
        resids = []
        for i in range(k):
            nb = self._x[i].shape[1]
            resids.append(self._y[i] - self._x[i] @ beta[loc : loc + nb])
            loc += nb
        resid = np.hstack(resids)

        individual = AttrDict()
        debiased = cov_config.get("debiased", False)
        for i in range(k):
            wy = wye = self._wy[i]
            w = self._w[i]
            cons = bool(self.has_constant.iloc[i])
            if cons:
                wc = np.ones_like(wy) * np.sqrt(w)
                wye = wy - wc @ lstsq(wc, wy, rcond=None)[0]
            total_ss = float(np.squeeze(wye.T @ wye))
            stats = self._common_indiv_results(
                i,
                beta,
                cov,
                eps,
                resid,
                "OLS",
                cov_type,
                cov_est,
                0,
                debiased,
                cons,
                total_ss,
            )
            key = self._eq_labels[i]
            individual[key] = stats

        # All equations use the same observations
        nobs = eps.shape[0]
        results = self._common_results(
            beta, cov, "OLS", 0, nobs, cov_type, sigma, individual, debiased
        )
        results["wresid"] = eps
        results["cov_estimator"] = cov_est
        results["cov_config"] = cov_est.cov_config
        individual = results["individual"]
        r2s = [individual[eq].r2 for eq in individual]
        results["system_r2"] = self._system_r2(
            resid, sigma, "ols", False, debiased, r2s
        )

        return SystemResults(results)

    @property
    def has_constant(self) -> Series:
        """Vector indicating which equations contain constants."""
        return self._has_constant

    def _f_stat(
        self, stats: AttrDict, debiased: bool
    ) -> WaldTestStatistic | InvalidTestStatistic:
        cov = stats.cov
        k = cov.shape[0]
        sel = list(range(k))
        if stats.has_constant:
            sel.pop(stats.constant_loc)
        cov = cov[sel][:, sel]
        params = stats.params[sel]
        df = params.shape[0]
        nobs = stats.nobs
        null = "All parameters ex. constant are zero"
        name = "Equation F-statistic"
        try:
            stat = float(np.squeeze(params.T @ inv(cov) @ params))

        except np.linalg.LinAlgError:
            return InvalidTestStatistic(
                "Covariance is singular, possibly due to constraints.", name=name
            )

        if debiased:
            total_reg = np.sum([s.shape[1] for s in self._wx])
            df_denom = len(self._wx) * nobs - total_reg
            wald = WaldTestStatistic(stat / df, null, df, df_denom=df_denom, name=name)
        else:
            return WaldTestStatistic(stat, null=null, df=df, name=name)

        return wald

    def _common_indiv_results(
        self,
        index: int,
        beta: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        wresid: linearmodels.typing.data.Float64Array,
        resid: linearmodels.typing.data.Float64Array,
        method: str,
        cov_type: str,
        cov_est: (
            HomoskedasticCovariance
            | HeteroskedasticCovariance
            | KernelCovariance
            | ClusteredCovariance
            | GMMHeteroskedasticCovariance
            | GMMHomoskedasticCovariance
        ),
        iter_count: int,
        debiased: bool,
        constant: bool,
        total_ss: float,
        *,
        weight_est: (
            HomoskedasticWeightMatrix
            | HeteroskedasticWeightMatrix
            | KernelWeightMatrix
            | None
        ) = None,
    ) -> AttrDict:
        loc = 0
        for i in range(index):
            loc += self._wx[i].shape[1]
        i = index
        stats = AttrDict()
        # Static properties
        stats["eq_label"] = self._eq_labels[i]
        stats["dependent"] = self._dependent[i].cols[0]
        stats["instruments"] = (
            self._instr[i].cols if self._instr[i].shape[1] > 0 else None
        )
        stats["endog"] = self._endog[i].cols if self._endog[i].shape[1] > 0 else None
        stats["method"] = method
        stats["cov_type"] = cov_type
        stats["cov_estimator"] = cov_est
        stats["cov_config"] = cov_est.cov_config
        stats["weight_estimator"] = weight_est
        stats["index"] = self._dependent[i].rows
        stats["original_index"] = self._original_index
        stats["iter"] = iter_count
        stats["debiased"] = debiased
        stats["has_constant"] = constant
        assert self._constant_loc is not None
        stats["constant_loc"] = self._constant_loc[i]

        # Parameters, errors and measures of fit
        wxi = self._wx[i]
        nobs, df = wxi.shape
        b = beta[loc : loc + df]
        e = wresid[:, [i]]
        nobs = e.shape[0]
        df_c = nobs - int(constant)
        df_r = nobs - df

        stats["params"] = b
        stats["cov"] = cov[loc : loc + df, loc : loc + df]
        stats["wresid"] = e
        stats["nobs"] = nobs
        stats["df_model"] = df
        # The residuals do not use the weights, but the sum of squares does
        weighted_resid = resid[:, [i]] * np.sqrt(self._w[i])
        stats["resid"] = resid[:, [i]]
        stats["weighted_resid"] = weighted_resid
        stats["fitted"] = self._x[i] @ b
        stats["resid_ss"] = float(np.squeeze(weighted_resid.T @ weighted_resid))
        stats["total_ss"] = total_ss
        # The size of the dependent variable, before the mean is removed, which sets
        # the rounding error in the total sum of squares
        wy = self._wy[i]
        y_ss = float(np.squeeze(wy.T @ wy))
        stats["r2"] = r_squared(stats.resid_ss, stats.total_ss, y_ss, nobs)
        if np.isnan(stats.r2):
            stats["r2a"] = np.nan
        else:
            stats["r2a"] = 1.0 - (stats.resid_ss / df_r) / (stats.total_ss / df_c)

        names = self._param_names[loc : loc + df]
        offset = len(stats.eq_label) + 1
        stats["param_names"] = [n[offset:] for n in names]

        # F-statistic
        stats["f_stat"] = self._f_stat(stats, debiased)

        return stats

    def _common_results(
        self,
        beta: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        method: str,
        iter_count: int,
        nobs: int,
        cov_type: str,
        sigma: linearmodels.typing.data.Float64Array,
        individual: AttrDict,
        debiased: bool,
    ) -> AttrDict:
        results = AttrDict()
        results["method"] = method
        results["iter"] = iter_count
        results["nobs"] = nobs
        results["cov_type"] = cov_type
        results["index"] = self._dependent[0].rows
        results["original_index"] = self._original_index
        names = list(individual.keys())
        results["sigma"] = DataFrame(sigma, columns=names, index=names)
        results["individual"] = individual
        results["params"] = beta
        results["df_model"] = beta.shape[0]
        results["param_names"] = self._param_names
        results["cov"] = cov
        results["debiased"] = debiased

        total_ss = resid_ss = 0.0
        residuals = []
        weighted_residuals = []
        for key in individual:
            total_ss += individual[key].total_ss
            resid_ss += individual[key].resid_ss
            residuals.append(individual[key].resid)
            weighted_residuals.append(individual[key].weighted_resid)
        resid = np.hstack(residuals)

        results["resid_ss"] = resid_ss
        results["total_ss"] = total_ss
        y_ss = sum(float(np.squeeze(wy.T @ wy)) for wy in self._wy)
        results["r2"] = r_squared(results.resid_ss, results.total_ss, y_ss, nobs)
        results["resid"] = resid
        results["weighted_resid"] = np.hstack(weighted_residuals)
        results["constraints"] = self._constraints
        results["model"] = self

        x = self._x
        k = len(x)
        loc = 0
        fitted_vals = []
        for i in range(k):
            nb = x[i].shape[1]
            b = beta[loc : loc + nb]
            fitted_vals.append(x[i] @ b)
            loc += nb
        fitted = np.hstack(fitted_vals)

        results["fitted"] = fitted

        return results

    def _system_r2(
        self,
        eps: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        method: str,
        full_cov: bool,
        debiased: bool,
        r2s: Sequence[float],
    ) -> Series:
        sigma_resid = sigma

        # System regression on a constant using weights if provided
        wy, w = self._wy, self._w
        wi = [
            cast("linearmodels.typing.data.Float64Array", np.sqrt(weights))
            for weights in w
        ]
        if method == "ols":
            est_sigma: linearmodels.typing.data.Float64Array = np.eye(len(wy))
        else:  # gls
            est_sigma = sigma
            if not full_cov:
                est_sigma = np.diag(np.diag(est_sigma))
        est_sigma_inv = cast("linearmodels.typing.data.Float64Array", inv(est_sigma))
        nobs = wy[0].shape[0]
        k = len(wy)
        xpx = blocked_inner_prod(wi, est_sigma_inv)
        xpy = np.zeros((k, 1))
        for i in range(k):
            sy = np.zeros((nobs, 1))
            for j in range(k):
                sy += est_sigma_inv[i, j] * wy[j]
            xpy[i : (i + 1)] = wi[i].T @ sy

        mu = _parameters_from_xprod(xpx, xpy)
        # eps are residuals that do not use the weights. The measures use the weighted
        # residuals of the model and of the regression on a constant
        eps = eps * np.hstack(wi)
        eps_const = np.hstack([wi[j] * (self._y[j] - mu[j]) for j in range(k)])
        # The size of the dependent variables, which sets the rounding error of the
        # total sums of squares, before the means are removed
        wy_all = np.hstack([wi[j] * self._y[j] for j in range(k)])
        y_ss = (wy_all**2).sum(0)
        # A dependent variable that does not vary has no total sum of squares. This is
        # found from its weighted mean, since the regression on a constant is GLS, and
        # its errors are large if the residuals of the equations are collinear, as they
        # are when a dependent variable does not vary
        own_ss = np.array(
            [
                ((wy[j] - wi[j] * ((w[j] * self._y[j]).sum() / w[j].sum())) ** 2).sum()
                for j in range(k)
            ]
        )
        varies = np.array(
            [
                not total_ss_is_zero(t, s, nobs)
                for t, s in zip(own_ss, y_ss, strict=True)
            ]
        )
        if not varies.any():
            return Series(
                {
                    "mcelroy": np.nan,
                    "berndt": np.nan,
                    "judge": np.nan,
                    "dhrymes": np.nan,
                }
            )
        # Judge
        judge = r_squared((eps**2).sum(), (eps_const**2).sum(), y_ss.sum(), nobs)
        # Dhrymes. The equations without a total sum of squares have no weight
        tot_eps_const_sq = (eps_const**2).sum(0)
        r2s_arr = np.asarray(r2s)
        dhrymes = (r2s_arr[varies] * tot_eps_const_sq[varies]).sum()
        dhrymes /= tot_eps_const_sq[varies].sum()

        # Berndt
        sigma_y = (eps_const.T @ eps_const / nobs) * self._sigma_scale(debiased)
        berndt = np.nan
        # Avoid division by 0. The matrix is singular if an equation has no variance
        if varies.all() and np.linalg.det(sigma_y) > 0:
            berndt = 1 - np.linalg.det(sigma_resid) / np.linalg.det(sigma_y)

        mcelroy = np.nan
        # Check that the matrix is invertible
        if np.linalg.matrix_rank(sigma) == sigma.shape[0]:
            # McElroy
            sigma_m12 = inv_matrix_sqrt(sigma)
            std_eps = eps @ sigma_m12
            numerator = (std_eps**2).sum()
            std_eps_const = eps_const @ sigma_m12
            denom = (std_eps_const**2).sum()
            std_y_ss = ((wy_all @ sigma_m12) ** 2).sum()
            mcelroy = r_squared(numerator, denom, std_y_ss, nobs)
        r2 = {"mcelroy": mcelroy, "berndt": berndt, "judge": judge, "dhrymes": dhrymes}
        return Series(r2)

    def _gls_finalize(
        self,
        beta: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        full_sigma: linearmodels.typing.data.Float64Array,
        est_sigma: linearmodels.typing.data.Float64Array,
        gls_eps: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
        full_cov: bool,
        cov_type: str,
        iter_count: int,
        **cov_config: bool,
    ) -> SystemResults:
        """Collect results to return after GLS estimation"""
        k = len(self._wy)

        # Covariance estimation
        cov_estimator = COV_EST[cov_type]
        gls_eps = np.reshape(gls_eps, (k, gls_eps.shape[0] // k)).T
        eps = np.reshape(eps, (k, eps.shape[0] // k)).T
        # The covariance combines the weighted regressors with the residuals, so the
        # residuals must be weighted as well
        weps = eps * np.sqrt(np.hstack(self._w))
        cov_est = cov_estimator(
            self._wxhat,
            weps,
            sigma,
            full_sigma,
            gls=True,
            constraints=self._constraints,
            **cov_config,
        )
        cov = cov_est.cov

        # Repackage results for individual equations
        individual = AttrDict()
        debiased = cov_config.get("debiased", False)
        method = "Iterative GLS" if iter_count > 1 else "GLS"
        for i in range(k):
            cons = bool(self.has_constant.iloc[i])

            if cons:
                c = np.sqrt(self._w[i])
                ye = self._wy[i] - c @ lstsq(c, self._wy[i], rcond=None)[0]
            else:
                ye = self._wy[i]
            total_ss = float(np.squeeze(ye.T @ ye))
            stats = self._common_indiv_results(
                i,
                beta,
                cov,
                gls_eps,
                eps,
                method,
                cov_type,
                cov_est,
                iter_count,
                debiased,
                cons,
                total_ss,
            )
            key = self._eq_labels[i]
            individual[key] = stats

        # Populate results dictionary. All equations use the same observations
        nobs = eps.shape[0]
        results = self._common_results(
            beta,
            cov,
            method,
            iter_count,
            nobs,
            cov_type,
            est_sigma,
            individual,
            debiased,
        )

        # wresid is different between GLS and OLS
        wresiduals = [
            individual[individual_key].wresid for individual_key in individual
        ]
        wresid = np.hstack(wresiduals)
        results["wresid"] = wresid
        results["cov_estimator"] = cov_est
        results["cov_config"] = cov_est.cov_config
        individual = results["individual"]
        r2s = [individual[eq].r2 for eq in individual]
        results["system_r2"] = self._system_r2(
            eps, sigma, "gls", full_cov, debiased, r2s
        )

        return SystemResults(results)

    def _sigma_scale(
        self, debiased: bool
    ) -> float | linearmodels.typing.data.Float64Array:
        if not debiased:
            return 1.0
        nobs = float(self._wx[0].shape[0])
        scales: linearmodels.typing.data.Float64Array = np.array(
            [nobs - x.shape[1] for x in self._wx], dtype=np.float64
        )
        scales = cast("linearmodels.typing.data.Float64Array", np.sqrt(nobs / scales))
        return scales[:, None] @ scales[None, :]

    @property
    def constraints(self) -> LinearConstraint | None:
        """
        Model constraints.

        Returns
        -------
        LinearConstraint or None
            Constraint object, which is None if the model has no constraints.
        """
        return self._constraints

    def add_constraints(self, r: DataFrame, q: Series | None = None) -> None:
        r"""
        Add parameter constraints to a model.

        Parameters
        ----------
        r : DataFrame
            Constraint matrix. nconstraints by nparameters.
        q : Series, optional
            Constraint values (nconstraints).  If not set, set to 0.

        Notes
        -----
        Constraints are of the form

        .. math ::

            r \beta = q

        The property `param_names` can be used to determine the order of
        parameters.
        """
        self._constraints = LinearConstraint(
            r, q=q, num_params=len(self._param_names), require_pandas=True
        )

    def reset_constraints(self) -> None:
        """Remove all model constraints."""
        self._constraints = None

    @property
    def param_names(self) -> list[str]:
        """
        Model parameter names.

        Returns
        -------
        list of str
            Normalized, unique model parameter names.
        """
        return self._param_names


class _LSSystemModelBase(_SystemModelBase):
    """Base class for least-squares-based system estimators."""

    def fit(
        self,
        *,
        method: Optional[Literal["ols", "gls"]] = None,
        full_cov: bool = True,
        iterate: bool = False,
        iter_limit: int = 100,
        tol: float = 1e-6,
        cov_type: str = "robust",
        **cov_config: bool,
    ) -> SystemResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        method : {"ols", "gls"}, optional
            Estimation method. The names are not case sensitive. If not provided, the
            method is selected based on the regressors, using OLS only if all
            regressors are identical and the model has no constraints, and GLS
            otherwise. The two names force the use of OLS or GLS.
        full_cov : bool, optional
            Flag indicating whether to utilize information in correlations
            when estimating the model with GLS. The default is True.
        iterate : bool, optional
            Flag indicating to iterate GLS until convergence or until iter_limit
            iterations have been completed. The default is False.
        iter_limit : int, optional
            Maximum number of iterations for iterative GLS. The default is 100.
        tol : float, optional
            Tolerance to use when checking for convergence in iterative GLS. The
            default is 1e-6.
        cov_type : str, optional
            Name of covariance estimator to use. The default is "robust". The names
            are not case sensitive. Supported covariance estimators are:

            * "unadjusted", "homoskedastic" - Classic covariance estimator
            * "robust", "heteroskedastic" - Heteroskedasticity robust
              covariance estimator
            * "kernel", "hac" - Allows for heteroskedasticity and autocorrelation
            * "clustered" - Allows for 1 and 2-way clustering of errors
              (Rogers).

        **cov_config : dict
            Additional parameters to pass to the covariance estimator. The keys
            that are used depend on ``cov_type``. All estimators accept ``debiased``
            (bool), which employs a small-sample adjustment and is False by
            default. "kernel" also accepts ``kernel`` (str), the name of the kernel,
            which is "bartlett" if it is not provided, and ``bandwidth`` (float),
            which is estimated from the data if it is not provided. The names of the
            kernels are not case sensitive. "clustered" also
            accepts ``clusters`` (array_like), the cluster assignment of each
            observation with one or two columns, and ``group_debias`` (bool), which
            debiases by the number of groups and is False by default. "unadjusted"
            and "robust" accept only ``debiased``.

        Returns
        -------
        SystemResults
            Estimation results.

        Raises
        ------
        ValueError
            If ``method`` is not "ols" or "gls", or if the labels of ``clusters``
            do not match the labels of the inputs.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 raised the ``TypeError`` of the constructor of the covariance
            estimator.
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.

        See Also
        --------
        linearmodels.system.covariance.HomoskedasticCovariance
            Homoskedastic covariance.
        linearmodels.system.covariance.HeteroskedasticCovariance
            Heteroskedasticity robust covariance.
        linearmodels.system.covariance.KernelCovariance
            Kernel (HAC) covariance.
        linearmodels.system.covariance.ClusteredCovariance
            Clustered covariance.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _LS_COV_KEYS, KERNEL_LOOKUP)
        self._check_clusters(cov_config)
        if method is None:
            method = (
                "ols" if (self._common_exog and self._constraints is None) else "gls"
            )
        else:
            if method.lower() not in ("ols", "gls"):
                raise ValueError(
                    f"method must be 'ols' or 'gls' when not None. Got {method}."
                )
            method = cast("Literal['ols', 'gls']", method.lower())

        cov_type = cov_type.lower()
        if cov_type not in COV_TYPES:
            raise unknown_option_error("cov_type", cov_type, COV_TYPES)
        cov_type = COV_TYPES[cov_type]
        k = len(self._dependent)
        col_sizes = [0] + [v.shape[1] for v in self._x]
        col_idx = [int(i) for i in np.cumsum(col_sizes)]
        total_cols = col_idx[-1]
        self._construct_xhat()
        beta, eps = self._multivariate_ls_fit()
        nobs = eps.shape[0]
        debiased = cov_config.get("debiased", False)
        full_sigma = sigma = (eps.T @ eps / nobs) * self._sigma_scale(debiased)

        if method == "ols":
            return self._multivariate_ls_finalize(
                beta, eps, sigma, cov_type, **cov_config
            )

        beta_hist = [beta]
        nobs = eps.shape[0]
        iter_count = 0
        delta = np.inf
        while (
            (iter_count < iter_limit and iterate) or iter_count == 0
        ) and delta >= tol:
            beta, eps, sigma, est_sigma = self._gls_estimate(
                eps, nobs, total_cols, col_idx, full_cov, debiased
            )
            beta_hist.append(beta)
            diff = beta_hist[-1] - beta_hist[-2]
            delta = float(np.sqrt(np.mean(diff**2)))
            iter_count += 1

        if iterate and self._sigma is None:
            # The covariance uses the residual covariance at the final estimates
            # rather than the one from the first step
            full_sigma = (eps.T @ eps / nobs) * self._sigma_scale(debiased)

        sigma_m12 = inv_matrix_sqrt(sigma)
        wy = blocked_column_product(self._wy, sigma_m12)
        wx = blocked_diag_product(self._wx, sigma_m12)
        gls_eps = wy - wx @ beta

        y = blocked_column_product(self._y, np.eye(k))
        x = blocked_diag_product(self._x, np.eye(k))
        eps = y - x @ beta

        return self._gls_finalize(
            beta,
            sigma,
            full_sigma,
            est_sigma,
            gls_eps,
            eps,
            full_cov,
            cov_type,
            iter_count,
            **cov_config,
        )


class IV3SLS(_LSSystemModelBase):
    r"""
    Three-stage Least Squares (3SLS) Estimator.

    Parameters
    ----------
    equations : dict of str to tuple or dict
        Dictionary-like structure containing dependent, exogenous, endogenous
        and instrumental variables.  Each key is an equations label and must
        be a string. Each value must be either a tuple of the form (dependent,
        exog, endog, instrument[, weights]) or a dictionary with keys "dependent",
        and at least one of "exog" or "endog" and "instruments".  When using a
        tuple, values must be provided for all 4 variables, although either
        empty arrays or `None` can be passed if a category of variable is not
        included in a model. The dictionary may contain optional keys for
        "exog", "endog", "instruments", and "weights". "exog" can be omitted
        if all variables in an equation are endogenous. Alternatively, "exog"
        can contain either an empty array or `None` to indicate that an
        equation contains no exogenous regressors. Similarly "endog" and
        "instruments" can either be omitted or may contain an empty array (or
        `None`) if all variables in an equation are exogenous. The variables are
        array_like, such as lists, NumPy arrays, pandas objects and xarray
        DataArrays, and a ``TypeError`` that names the variable is raised for
        complex values or text.
    sigma : array_like, optional
        Prespecified residual covariance to use in GLS estimation, with shape
        (nequations, nequations). If not provided, FGLS is implemented based on an
        estimate of sigma.

    Notes
    -----
    Estimates a set of regressions which are seemingly unrelated in the sense
    that separate estimation would lead to consistent parameter estimates.
    Each equation is of the form

    .. math::

        y_{i,k} = x_{i,k}\beta_i + \epsilon_{i,k}

    where k denotes the equation and i denoted the observation index. By
    stacking vertically arrays of dependent and placing the exogenous
    variables into a block diagonal array, the entire system can be compactly
    expressed as

    .. math::

        Y = X\beta + \epsilon

    where

    .. math::

        Y = \left[\begin{array}{c}Y_1 \\ Y_2 \\ \vdots \\ Y_K\end{array}\right]

    and

    .. math::

        X = \left[\begin{array}{cccc}
                 X_1 & 0 & \ldots & 0 \\
                 0 & X_2 & \dots & 0 \\
                 \vdots & \vdots & \ddots & \vdots \\
                 0 & 0 & \dots & X_K
            \end{array}\right]

    The system instrumental variable (IV) estimator is

    .. math::

        \hat{\beta}_{IV} & = (X'Z(Z'Z)^{-1}Z'X)^{-1}X'Z(Z'Z)^{-1}Z'Y \\
                         & = (\hat{X}'\hat{X})^{-1}\hat{X}'Y

    where :math:`\hat{X} = Z(Z'Z)^{-1}Z'X`.  When certain conditions are
    satisfied, a GLS estimator of the form

    .. math::

        \hat{\beta}_{3SLS} = (\hat{X}'\Omega^{-1}\hat{X})^{-1}\hat{X}'\Omega^{-1}Y

    can improve accuracy of coefficient estimates where

    .. math::

        \Omega = \Sigma \otimes I_N

    where :math:`\Sigma` is the covariance matrix of the residuals.

    The weights of an equation, if they are given, must be strictly positive and
    finite. They are rescaled to have a mean of 1 in the equation. A NaN weight is
    missing, and its observation is dropped from all of the equations.

    Rows are paired by position, both within an equation and across the
    equations. An input that carries labels (a pandas Series or DataFrame
    with an index other than the default 0, 1, ..., n-1, or an xarray
    DataArray with coordinates) must use the same index, in the same order,
    as the dependent variable of its equation, so that a value cannot be
    attached to the wrong observation. The dependent variables that carry
    labels must also use the same index in every equation. A ``ValueError``
    is raised if an index differs. Either reindex the input to the index of
    the dependent variable, or call ``to_numpy()`` to pair its values by
    position. NumPy arrays and pandas objects with a default index do not
    carry labels and are always paired by position. If none of the dependent
    variables carries labels, the inputs that do must use the same index as the
    first of them that does, in the order ``exog``, ``endog``, ``instruments``
    and ``weights`` of each equation.
    The same rule applies to ``clusters`` passed to ``fit``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> None:
        super().__init__(equations, sigma=sigma)

    @classmethod
    def multivariate_iv(
        cls,
        dependent: linearmodels.typing.data.ArrayLike,
        exog: linearmodels.typing.data.ArrayLike | None = None,
        endog: linearmodels.typing.data.ArrayLike | None = None,
        instruments: linearmodels.typing.data.ArrayLike | None = None,
    ) -> IV3SLS:
        """
        Interface for specification of multivariate IV models.

        Parameters
        ----------
        dependent : array_like
            Array of dependent variables with shape (nobs, ndep).
        exog : array_like, optional
            Array of exogenous regressors common to all models with shape
            (nobs, nexog).
        endog : array_like, optional
            Array of endogenous regressors common to all models with shape
            (nobs, nendog).
        instruments : array_like, optional
            Array of instruments to use in all equations with shape (nobs, ninstr).

        Returns
        -------
        IV3SLS
            Model instance.

        Raises
        ------
        ValueError
            If neither ``exog`` nor ``endog`` is provided.

        Notes
        -----
        At least one of exog or endog must be provided.

        Utility function to simplify the construction of multivariate IV
        models which all use the same regressors and instruments. Constructs
        the dictionary of equations from the variables using the common
        exogenous, endogenous and instrumental variables.
        """
        equations = {}
        dependent_ivd = IVData(dependent, var_name="dependent")
        if exog is None and endog is None:
            raise ValueError("At least one of exog or endog must be provided")
        nobs = dependent_ivd.shape[0]
        exog_ivd = IVData(exog, var_name="exog", nobs=nobs)
        endog_ivd = IVData(endog, var_name="endog", nobs=nobs)
        instr_ivd = IVData(instruments, var_name="instruments", nobs=nobs)
        for col in dependent_ivd.pandas:
            equations[str(col)] = {
                # TODO: Bug in pandas-stubs
                #  https://github.com/pandas-dev/pandas-stubs/issues/97
                "dependent": dependent_ivd.pandas[[col]],
                "exog": exog_ivd.pandas,
                "endog": endog_ivd.pandas,
                "instruments": instr_ivd.pandas,
            }

        return cls(equations)

    @classmethod
    def from_formula(
        cls,
        formula: str | dict[str, str],
        data: DataFrame,
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        weights: Mapping[str, linearmodels.typing.data.ArrayLike] | None = None,
    ) -> IV3SLS:
        """
        Specify a 3SLS using the formula interface.

        Parameters
        ----------
        formula : str or dict of str to str
            Either a string or a dictionary of strings where each value in
            the dictionary represents a single equation. See Notes for a
            description of the accepted syntax.
        data : DataFrame
            Frame containing named variables.
        sigma : array_like, optional
            Prespecified residual covariance to use in GLS estimation, with shape
            (nequations, nequations). If not provided, FGLS is implemented based on
            an estimate of sigma.
        weights : dict of str to array_like, optional
            Dictionary like object (e.g. a DataFrame) containing variable
            weights.  Each entry must have the same number of observations as
            data.  If an equation label is not a key weights, the weights will
            be set to unity.
            The weights of an equation must be strictly positive and finite. They are
            rescaled to have a mean of 1 in the equation. A NaN weight is missing, and
            its observation is dropped from all of the equations.

        Returns
        -------
        IV3SLS
            Model instance.

        Notes
        -----
        Models can be specified in one of two ways. The first uses curly
        braces to encapsulate equations.  The second uses a dictionary
        where each key is an equation name.

        Examples
        --------
        The simplest format uses standard formulas for each equation
        in a dictionary.  Best practice is to use an Ordered Dictionary

        >>> import pandas as pd
        >>> import numpy as np
        >>> cols = ["y1", "x1_1", "x1_2", "z1", "y2", "x2_1", "x2_2", "z2"]
        >>> data = pd.DataFrame(np.random.randn(500, 8), columns=cols)
        >>> from linearmodels.system import IV3SLS
        >>> formula = {"eq1": "y1 ~ 1 + x1_1 + [x1_2 ~ z1]",
        ...            "eq2": "y2 ~ 1 + x2_1 + [x2_2 ~ z2]"}
        >>> mod = IV3SLS.from_formula(formula, data)

        The second format uses curly braces {} to surround distinct equations

        >>> formula = "{y1 ~ 1 + x1_1 + [x1_2 ~ z1]} {y2 ~ 1 + x2_1 + [x2_2 ~ z2]}"
        >>> mod = IV3SLS.from_formula(formula, data)

        It is also possible to include equation labels when using curly braces

        >>> formula = "{eq1: y1 ~ x1_1 + [x1_2 ~ z1]} {eq2: y2 ~ 1 + [x2_2 ~ z2]}"
        >>> mod = IV3SLS.from_formula(formula, data)
        """
        context = capture_context(1)
        parser = SystemFormulaParser(formula, data, weights, context=context)
        eqns = parser.data
        mod = cls(eqns, sigma=sigma)
        mod.formula = formula
        return mod


class SUR(_LSSystemModelBase):
    r"""
    Seemingly unrelated regression estimation (SUR/SURE).

    Parameters
    ----------
    equations : dict of str to tuple or dict
        Dictionary-like structure containing dependent and exogenous variable
        values.  Each key is an equations label and must be a string. Each
        value must be either a tuple of the form (dependent,
        exog, [weights]) or a dictionary with keys "dependent" and "exog" and
        the optional key "weights". The variables are array_like, such as lists,
        NumPy arrays, pandas objects and xarray DataArrays, and a ``TypeError``
        that names the variable is raised for complex values or text.
    sigma : array_like, optional
        Prespecified residual covariance to use in GLS estimation, with shape
        (nequations, nequations). If not provided, FGLS is implemented based on an
        estimate of sigma.

    Notes
    -----
    Estimates a set of regressions which are seemingly unrelated in the sense
    that separate estimation would lead to consistent parameter estimates.
    Each equation is of the form

    .. math::

        y_{i,k} = x_{i,k}\beta_i + \epsilon_{i,k}

    where k denotes the equation and i denoted the observation index. By
    stacking vertically arrays of dependent and placing the exogenous
    variables into a block diagonal array, the entire system can be compactly
    expressed as

    .. math::

        Y = X\beta + \epsilon

    where

    .. math::

        Y = \left[\begin{array}{c}Y_1 \\ Y_2 \\ \vdots \\ Y_K\end{array}\right]

    and

    .. math::

        X = \left[\begin{array}{cccc}
                 X_1 & 0 & \ldots & 0 \\
                 0 & X_2 & \dots & 0 \\
                 \vdots & \vdots & \ddots & \vdots \\
                 0 & 0 & \dots & X_K
            \end{array}\right]

    The system OLS estimator is

    .. math::

        \hat{\beta}_{OLS} = (X'X)^{-1}X'Y

    When certain conditions are satisfied, a GLS estimator of the form

    .. math::

        \hat{\beta}_{GLS} = (X'\Omega^{-1}X)^{-1}X'\Omega^{-1}Y

    can improve accuracy of coefficient estimates where

    .. math::

        \Omega = \Sigma \otimes I_N

    where :math:`\Sigma` is the covariance matrix of the residuals.

    SUR is a special case of 3SLS where there are no endogenous regressors and
    no instruments.

    The weights of an equation, if they are given, must be strictly positive and
    finite. They are rescaled to have a mean of 1 in the equation. A NaN weight is
    missing, and its observation is dropped from all of the equations.

    Rows are paired by position, both within an equation and across the
    equations. An input that carries labels (a pandas Series or DataFrame
    with an index other than the default 0, 1, ..., n-1, or an xarray
    DataArray with coordinates) must use the same index, in the same order,
    as the dependent variable of its equation, so that a value cannot be
    attached to the wrong observation. The dependent variables that carry
    labels must also use the same index in every equation. A ``ValueError``
    is raised if an index differs. Either reindex the input to the index of
    the dependent variable, or call ``to_numpy()`` to pair its values by
    position. NumPy arrays and pandas objects with a default index do not
    carry labels and are always paired by position. If none of the dependent
    variables carries labels, the inputs that do must use the same index as the
    first of them that does, in the order ``exog``, ``endog``, ``instruments``
    and ``weights`` of each equation.
    The same rule applies to ``clusters`` passed to ``fit``.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> None:
        if not isinstance(equations, Mapping):
            raise TypeError("equations must be a dictionary-like")
        for key in equations:
            if not isinstance(key, str):
                raise ValueError("Equation labels (keys) must be strings")
        reformatted = {}
        for key in equations:
            eqn = equations[key]
            if isinstance(eqn, tuple):
                if len(eqn) == 3:
                    w = eqn[-1]
                    eqn = eqn[:2]
                    eqn = (*eqn, None, None, w)
                else:
                    eqn = (*eqn, None, None)
            reformatted[key] = eqn
        super().__init__(reformatted, sigma=sigma)
        self._model_name = "Seemingly Unrelated Regression (SUR)"

    @classmethod
    def multivariate_ls(
        cls,
        dependent: linearmodels.typing.data.ArrayLike,
        exog: linearmodels.typing.data.ArrayLike,
    ) -> SUR:
        """
        Interface for specification of multivariate regression models.

        Parameters
        ----------
        dependent : array_like
            Array of dependent variables with shape (nobs, ndep).
        exog : array_like
            Array of exogenous regressors common to all models with shape
            (nobs, nvar).

        Returns
        -------
        SUR
            Model instance.

        Notes
        -----
        Utility function to simplify the construction of multivariate
        regression models which all use the same regressors. Constructs
        the dictionary of equations from the variables using the common
        exogenous variable.

        Examples
        --------
        A simple CAP-M can be estimated as a multivariate regression

        >>> from linearmodels.datasets import french
        >>> from linearmodels.system import SUR
        >>> data = french.load()
        >>> portfolios = data[["S1V1","S1V5","S5V1","S5V5"]]
        >>> factors = data[["MktRF"]].copy()
        >>> factors["alpha"] = 1
        >>> mod = SUR.multivariate_ls(portfolios, factors)
        """
        equations = {}
        dependent_ivd = IVData(dependent, var_name="dependent")
        exog_ivd = IVData(exog, var_name="exog")
        for col in dependent_ivd.pandas:
            # TODO: Bug in pandas-stubs
            #  https://github.com/pandas-dev/pandas-stubs/issues/97
            equations[str(col)] = (dependent_ivd.pandas[[col]], exog_ivd.pandas)
        return cls(equations)

    @classmethod
    def from_formula(
        cls,
        formula: str | dict[str, str],
        data: DataFrame,
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        weights: Mapping[str, linearmodels.typing.data.ArrayLike] | None = None,
    ) -> SUR:
        """
        Specify a SUR using the formula interface.

        Parameters
        ----------
        formula : str or dict of str to str
            Either a string or a dictionary of strings where each value in
            the dictionary represents a single equation. See Notes for a
            description of the accepted syntax.
        data : DataFrame
            Frame containing named variables.
        sigma : array_like, optional
            Prespecified residual covariance to use in GLS estimation, with shape
            (nequations, nequations). If not provided, FGLS is implemented based on
            an estimate of sigma.
        weights : dict of str to array_like, optional
            Dictionary like object (e.g. a DataFrame) containing variable
            weights.  Each entry must have the same number of observations as
            data.  If an equation label is not a key weights, the weights will
            be set to unity.
            The weights of an equation must be strictly positive and finite. They are
            rescaled to have a mean of 1 in the equation. A NaN weight is missing, and
            its observation is dropped from all of the equations.

        Returns
        -------
        SUR
            Model instance.

        Notes
        -----
        Models can be specified in one of two ways. The first uses curly
        braces to encapsulate equations.  The second uses a dictionary
        where each key is an equation name.

        Examples
        --------
        The simplest format uses standard formulas for each equation
        in a dictionary.  Best practice is to use an Ordered Dictionary

        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame(np.random.randn(500, 4),
        ...                     columns=["y1", "x1_1", "y2", "x2_1"])
        >>> from linearmodels.system import SUR
        >>> formula = {"eq1": "y1 ~ 1 + x1_1", "eq2": "y2 ~ 1 + x2_1"}
        >>> mod = SUR.from_formula(formula, data)

        The second format uses curly braces {} to surround distinct equations

        >>> formula = "{y1 ~ 1 + x1_1} {y2 ~ 1 + x2_1}"
        >>> mod = SUR.from_formula(formula, data)

        It is also possible to include equation labels when using curly braces

        >>> formula = "{eq1: y1 ~ 1 + x1_1} {eq2: y2 ~ 1 + x2_1}"
        >>> mod = SUR.from_formula(formula, data)
        """
        context = capture_context(1)
        parser = SystemFormulaParser(formula, data, weights, context=context)
        eqns = parser.data
        mod = cls(eqns, sigma=sigma)
        mod.formula = formula
        return mod


class IVSystemGMM(_SystemModelBase):
    r"""
    System Generalized Method of Moments (GMM) estimation of linear IV models.

    Parameters
    ----------
    equations : dict of str to tuple or dict
        Dictionary-like structure containing dependent, exogenous, endogenous
        and instrumental variables.  Each key is an equations label and must
        be a string. Each value must be either a tuple of the form (dependent,
        exog, endog, instrument[, weights]) or a dictionary with keys "dependent",
        "exog".  The dictionary may contain optional keys for "endog",
        "instruments", and "weights". Endogenous and/or Instrument can be empty
        if all variables in an equation are exogenous. The variables are
        array_like, such as lists, NumPy arrays, pandas objects and xarray
        DataArrays, and a ``TypeError`` that names the variable is raised for
        complex values or text.
    sigma : array_like, optional
        Prespecified residual covariance to use in GLS estimation, with shape
        (nequations, nequations). If not provided, FGLS is implemented based on an
        estimate of sigma. Only used if weight_type is "unadjusted" or
        "homoskedastic". A ``UserWarning`` is issued, and sigma is ignored, if it is
        provided with another weight type.
    weight_type : str, optional
        Name of moment condition weight function to use in the GMM estimation. The
        default is "robust". The names are not case sensitive. The supported weight
        functions are:

        * "unadjusted", "homoskedastic" - Assume moments are homoskedastic
        * "robust", "heteroskedastic" - Allow for heteroskedasticity
        * "kernel", "hac" - Allow for heteroskedasticity and autocorrelation

    **weight_config : dict
        Additional keyword arguments to pass to the moment condition weight
        function. All weight types accept ``center`` (bool), whether to center the
        moment conditions, and ``debiased`` (bool), whether to use small-sample
        adjustments. Both are False by default. "kernel" also accepts ``kernel``
        (str), the name of the kernel, which is "bartlett" if it is not provided,
        ``bandwidth`` (float), and ``optimal_bw`` (bool). The names of the kernels
        are not case sensitive. See
        :class:`~linearmodels.system.gmm.HomoskedasticWeightMatrix`,
        :class:`~linearmodels.system.gmm.HeteroskedasticWeightMatrix` and
        :class:`~linearmodels.system.gmm.KernelWeightMatrix` for the details.

    Raises
    ------
    UnknownOptionError
        If ``weight_type`` is not supported or ``kernel`` is not a supported
        kernel. It is both a ``KeyError`` and a ``ValueError``.

    Notes
    -----
    Estimates a linear model using GMM. Each equation is of the form

    .. math::

        y_{i,k} = x_{i,k}\beta_i + \epsilon_{i,k}

    where k denotes the equation and i denoted the observation index. By
    stacking vertically arrays of dependent and placing the exogenous
    variables into a block diagonal array, the entire system can be compactly
    expressed as

    .. math::

        Y = X\beta + \epsilon

    where

    .. math::

        Y = \left[\begin{array}{c}Y_1 \\ Y_2 \\ \vdots \\ Y_K\end{array}\right]

    and

    .. math::

        X = \left[\begin{array}{cccc}
                 X_1 & 0 & \ldots & 0 \\
                 0 & X_2 & \dots & 0 \\
                 \vdots & \vdots & \ddots & \vdots \\
                 0 & 0 & \dots & X_K
            \end{array}\right]

    The system GMM estimator uses the moment condition

    .. math::

        z_{ij}(y_{ij} - x_{ij}\beta_j) = 0

    where j indexes the equation. The estimator for the coefficients is given
    by

    .. math::

        \hat{\beta}_{GMM} & = (X'ZW^{-1}Z'X)^{-1}X'ZW^{-1}Z'Y \\

    where :math:`W` is a positive definite weighting matrix.

    The weight matrix is the inverse of an estimate of the covariance of the
    moment conditions. ``weight_type`` selects the estimator: "unadjusted"
    ("homoskedastic"), "robust" ("heteroskedastic") or "kernel" (``"hac"``), and
    ``weight_config`` contains its options. These are ``center`` and
    ``debiased`` for all of them, and ``kernel``, ``bandwidth`` and
    ``optimal_bw`` for "kernel".

    The bandwidth of a kernel weight matrix (``weight_type="kernel"``) is
    ``bandwidth`` if it is given in ``weight_config``, and ``nobs - 2`` if
    ``optimal_bw`` is ``False``. Otherwise, which is the default and what
    ``optimal_bw=True`` does, it is estimated from the data, from the
    residuals that the weight matrix is computed from. These are the residuals
    of the first step estimates in two-step GMM, which is the default. A
    bandwidth that is estimated from the data, for the weight matrix or the
    covariance, needs at least 4 observations with the Bartlett kernel, 5 with
    the Parzen kernel and 6 with the quadratic spectral kernel. ``fit`` raises a
    ``ValueError`` for fewer observations, and if the estimated bandwidth is
    not finite. Pass ``bandwidth`` to use the kernel with fewer observations.
    See :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`. In 7.0
    and earlier the default was ``nobs - 2``, which is not a consistent
    estimator of the covariance of the moment conditions, so the estimates,
    standard errors and J statistic can be different. ``fit`` issues a
    ``DefaultBandwidthWarning`` (a ``UserWarning``) when it uses the default.
    To restore the previous behavior, pass ``optimal_bw=False`` or
    ``bandwidth=nobs - 2``, where ``nobs`` is the number of observations of
    each equation, in ``weight_config``, for example
    ``IVSystemGMM(equations, weight_type="kernel", optimal_bw=False)``. Passing
    ``bandwidth`` or ``optimal_bw`` silences the warning, and
    ``optimal_bw=True`` selects the bandwidth from the data without it. The
    kernel covariance of the parameters (``cov_type="kernel"`` in ``fit``) is
    not affected, since it has always estimated its bandwidth from the data
    unless ``bandwidth`` is given. See
    :class:`~linearmodels.system.gmm.KernelWeightMatrix`.

    The weights of an equation, if they are given, must be strictly positive and
    finite. They are rescaled to have a mean of 1 in the equation. A NaN weight is
    missing, and its observation is dropped from all of the equations.

    Rows are paired by position, both within an equation and across the
    equations. An input that carries labels (a pandas Series or DataFrame
    with an index other than the default 0, 1, ..., n-1, or an xarray
    DataArray with coordinates) must use the same index, in the same order,
    as the dependent variable of its equation, so that a value cannot be
    attached to the wrong observation. The dependent variables that carry
    labels must also use the same index in every equation. A ``ValueError``
    is raised if an index differs. Either reindex the input to the index of
    the dependent variable, or call ``to_numpy()`` to pair its values by
    position. NumPy arrays and pandas objects with a default index do not
    carry labels and are always paired by position. If none of the dependent
    variables carries labels, the inputs that do must use the same index as the
    first of them that does, in the order ``exog``, ``endog``, ``instruments``
    and ``weights`` of each equation.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        weight_type: str = "robust",
        **weight_config: bool | str | float,
    ) -> None:
        super().__init__(equations, sigma=sigma)
        weight_type = lower_option(weight_type)
        self._weight_type = weight_type
        self._weight_config = weight_config

        if weight_type not in GMM_TYPES:
            raise unknown_option_error("weight_type", weight_type, GMM_TYPES)

        if weight_type not in ("unadjusted", "homoskedastic") and sigma is not None:
            warnings.warn(
                "sigma has been provided but the estimated weight "
                "matrix not unadjusted (homoskedastic).  sigma will "
                "be ignored.",
                UserWarning,
                stacklevel=1,
            )
        weight_type = COV_TYPES[weight_type]
        # Kernel weights without a bandwidth select it from the data. fit warns
        # about the change of this default, and makes the choice explicit so that
        # the weight estimator does not warn as well
        self._default_weight_bandwidth = (
            weight_type == "kernel" and _uses_default_bandwidth(weight_config)
        )
        if self._default_weight_bandwidth:
            weight_config = {**weight_config, "optimal_bw": True}
        # Only the kernel weight matrix takes kernel, bandwidth and optimal_bw
        self._weight_est: HomoskedasticWeightMatrix
        if weight_type == "kernel":
            self._weight_est = KernelWeightMatrix(
                **cast("_KernelWeightConfig", weight_config)
            )
        else:
            self._weight_est = GMM_W_EST[weight_type](
                **cast("_WeightConfig", weight_config)
            )

    def fit(
        self,
        *,
        iter_limit: int = 2,
        tol: float = 1e-6,
        initial_weight: linearmodels.typing.data.Float64Array | None = None,
        cov_type: str = "robust",
        **cov_config: bool | float | str,
    ) -> GMMSystemResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        iter_limit : int, optional
            Maximum number of iterations for iterative GLS. The default is 2.
        tol : float, optional
            Tolerance to use when checking for convergence in iterative GLS. The
            default is 1e-6.
        initial_weight : ndarray, optional
            Initial weighting matrix to use in the first step. If not
            specified, uses the average outer-product of the set containing
            the exogenous variables and instruments.
        cov_type : str, optional
            Name of covariance estimator to use. The default is "robust". The names
            are not case sensitive. Supported covariance estimators are:

            * "unadjusted", "homoskedastic" - Classic covariance estimator
            * "robust", "heteroskedastic" - Heteroskedasticity robust
              covariance estimator
            * "kernel", "hac" - Heteroskedasticity and autocorrelation
              robust covariance estimator

        **cov_config : dict
            Additional parameters to pass to the covariance estimator. All
            estimators accept ``debiased`` (bool), which employs a small-sample
            adjustment and is False by default. "kernel" also accepts ``kernel``
            (str), the name of the kernel, which is "bartlett" if it is not
            provided, and ``bandwidth`` (float), which is estimated from the data
            if it is not given. The names of the kernels are not case sensitive.

        Returns
        -------
        GMMSystemResults
            Estimation results.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.
        ValueError
            If a bandwidth is estimated from the data, for the kernel weight
            matrix or for the kernel covariance, and there are fewer
            observations than the kernel needs: 4 for the Bartlett kernel, 5
            for the Parzen kernel and 6 for the quadratic spectral kernel. It
            is also raised if the estimated bandwidth is not finite. Pass
            ``bandwidth`` to use the kernel with fewer observations.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 raised the ``TypeError`` of the constructor of the covariance
            estimator.

        Notes
        -----
        If the model has a kernel weight matrix (``weight_type="kernel"``) and
        neither ``bandwidth`` nor ``optimal_bw`` was given when it was created,
        the bandwidth of the weight matrix is estimated from the data. In 7.0
        and earlier it was ``nobs - 2``. ``fit`` issues a
        ``DefaultBandwidthWarning`` (a ``UserWarning``) once for each call when
        the model uses the default bandwidth. See the notes of
        :class:`~linearmodels.system.model.IVSystemGMM` for how to restore the
        previous bandwidth.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _GMM_COV_KEYS, KERNEL_LOOKUP)
        if cov_type not in GMM_TYPES:
            raise unknown_option_error("cov_type", cov_type, GMM_TYPES)
        # Parameter estimation
        wx, wy, wz = self._wx, self._wy, self._wz
        k = len(wx)
        nobs = wx[0].shape[0]
        k_total = sum(a.shape[1] for a in wz)
        if self._default_weight_bandwidth:
            # Once for each call to fit, however many iterations it needs. The
            # warning refers to the call to fit
            _warn_default_bandwidth(
                weights=True, covariance=False, stacklevel=3, model="IVSystemGMM"
            )
        if initial_weight is None:
            w = blocked_inner_prod(wz, np.eye(k_total)) / nobs
        else:
            w = initial_weight
        assert w is not None
        beta_last = beta = self._blocked_gmm(
            wx,
            wy,
            wz,
            w=cast("linearmodels.typing.data.Float64Array", w),
            constraints=self.constraints,
        )
        _eps = []
        loc = 0
        for i in range(k):
            nb = wx[i].shape[1]
            b = beta[loc : loc + nb]
            _eps.append(wy[i] - wx[i] @ b)
            loc += nb
        eps = np.hstack(_eps)
        sigma = self._weight_est.sigma(eps, wx) if self._sigma is None else self._sigma
        vinv = None
        iters = 1
        norm = 10 * tol + 1
        while iters < iter_limit and norm > tol:
            sigma = (
                self._weight_est.sigma(eps, wx) if self._sigma is None else self._sigma
            )
            w = self._weight_est.weight_matrix(wx, wz, eps, sigma=sigma)
            beta = self._blocked_gmm(
                wx,
                wy,
                wz,
                w=cast("linearmodels.typing.data.Float64Array", w),
                constraints=self.constraints,
            )
            delta = beta_last - beta
            if vinv is None:
                winv = np.linalg.inv(w)
                xpz = blocked_cross_prod(wx, wz, np.eye(k))
                xpz = cast("linearmodels.typing.data.Float64Array", xpz / nobs)
                v = (xpz @ winv @ xpz.T) / nobs
                vinv = inv(v)
            norm = float(np.squeeze(delta.T @ vinv @ delta))
            beta_last = beta

            _eps = []
            loc = 0
            for i in range(k):
                nb = wx[i].shape[1]
                b = beta[loc : loc + nb]
                _eps.append(wy[i] - wx[i] @ b)
                loc += nb
            eps = np.hstack(_eps)
            iters += 1

        cov_type = COV_TYPES[cov_type]
        # Only the kernel covariance takes kernel and bandwidth
        cov: GMMHomoskedasticCovariance
        if cov_type == "kernel":
            cov = GMMKernelCovariance(
                wx,
                wz,
                eps,
                w,
                sigma=sigma,
                constraints=self._constraints,
                **cast("_GMMKernelCovConfig", cov_config),
            )
        else:
            cov = GMM_COV_EST[cov_type](
                wx,
                wz,
                eps,
                w,
                sigma=sigma,
                constraints=self._constraints,
                **cast("_GMMCovConfig", cov_config),
            )

        weps = eps
        _eps = []
        loc = 0
        x, y = self._x, self._y
        for i in range(k):
            nb = x[i].shape[1]
            b = beta[loc : loc + nb]
            _eps.append(y[i] - x[i] @ b)
            loc += nb
        eps = np.hstack(_eps)
        iters += 1
        return self._finalize_results(
            beta,
            cov.cov,
            weps,
            eps,
            cast("np.ndarray", w),
            sigma,
            iters - 1,
            cov_type,
            cov_config,
            cov,
        )

    @staticmethod
    def _blocked_gmm(
        x: linearmodels.typing.ArraySequence,
        y: linearmodels.typing.ArraySequence,
        z: linearmodels.typing.ArraySequence,
        *,
        w: linearmodels.typing.data.Float64Array,
        constraints: LinearConstraint | None = None,
    ) -> linearmodels.typing.data.Float64Array:
        k = len(x)
        xpz = blocked_cross_prod(x, z, np.eye(k))
        wi = np.linalg.inv(w)
        xpz_wi_zpx = xpz @ wi @ xpz.T
        zpy_arrs = []
        zpy_arrs = [(z[i].T @ y[i]) for i in range(k)]
        zpy = np.vstack(zpy_arrs)
        xpz_wi_zpy = xpz @ wi @ zpy
        params = _parameters_from_xprod(xpz_wi_zpx, xpz_wi_zpy, constraints=constraints)

        return params

    def _finalize_results(
        self,
        beta: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        weps: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
        wmat: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        iter_count: int,
        cov_type: str,
        cov_config: dict[str, bool | float | str],
        cov_est: GMMHeteroskedasticCovariance | GMMHomoskedasticCovariance,
    ) -> GMMSystemResults:
        """Collect results to return after GLS estimation"""
        k = len(self._wy)
        # Repackage results for individual equations
        individual = AttrDict()
        debiased = bool(cov_config.get("debiased", False))
        method = f"{iter_count}-Step System GMM"
        if iter_count > 2:
            method = "Iterative System GMM"
        for i in range(k):
            cons = bool(self.has_constant.iloc[i])

            if cons:
                c = np.sqrt(self._w[i])
                ye = self._wy[i] - c @ lstsq(c, self._wy[i], rcond=None)[0]
            else:
                ye = self._wy[i]
            total_ss = float(np.squeeze(ye.T @ ye))
            stats = self._common_indiv_results(
                i,
                beta,
                cov,
                weps,
                eps,
                method,
                cov_type,
                cov_est,
                iter_count,
                debiased,
                cons,
                total_ss,
                weight_est=self._weight_est,
            )

            key = self._eq_labels[i]
            individual[key] = stats

        # Populate results dictionary. All equations use the same observations
        nobs = eps.shape[0]
        results = self._common_results(
            beta, cov, method, iter_count, nobs, cov_type, sigma, individual, debiased
        )

        # wresid is different between GLS and OLS
        wresiduals = [
            individual[individual_key].wresid for individual_key in individual
        ]
        wresid = np.hstack(wresiduals)
        results["wresid"] = wresid
        results["wmat"] = wmat
        results["weight_type"] = self._weight_type
        results["weight_config"] = self._weight_est.config
        results["cov_estimator"] = cov_est
        results["cov_config"] = cov_est.cov_config
        results["weight_estimator"] = self._weight_est
        results["j_stat"] = self._j_statistic(beta, wmat)
        r2s = [individual[eq].r2 for eq in individual]
        results["system_r2"] = self._system_r2(eps, sigma, "gls", False, debiased, r2s)

        return GMMSystemResults(results)

    @classmethod
    def from_formula(
        cls,
        formula: str | dict[str, str],
        data: DataFrame,
        *,
        weights: dict[str, linearmodels.typing.data.ArrayLike] | None = None,
        weight_type: str = "robust",
        **weight_config: bool | str | float,
    ) -> IVSystemGMM:
        """
        Specify a system GMM model using the formula interface.

        Parameters
        ----------
        formula : str or dict of str to str
            Either a string or a dictionary of strings where each value in
            the dictionary represents a single equation. See Notes for a
            description of the accepted syntax.
        data : DataFrame
            Frame containing named variables.
        weights : dict of str to array_like, optional
            Dictionary like object (e.g. a DataFrame) containing variable
            weights.  Each entry must have the same number of observations as
            data.  If an equation label is not a key weights, the weights will
            be set to unity.
            The weights of an equation must be strictly positive and finite. They are
            rescaled to have a mean of 1 in the equation. A NaN weight is missing, and
            its observation is dropped from all of the equations.
        weight_type : str, optional
            Name of moment condition weight function to use in the GMM
            estimation. The default is "robust". The names are not case sensitive.
            The supported weight functions are:

            * "unadjusted", "homoskedastic" - Assume moments are homoskedastic
            * "robust", "heteroskedastic" - Allow for heteroskedasticity
            * "kernel", "hac" - Allow for heteroskedasticity and autocorrelation

        **weight_config : dict
            Additional keyword arguments to pass to the moment condition weight
            function. All weight types accept ``center`` (bool) and ``debiased``
            (bool), which are False by default. "kernel" also accepts ``kernel``
            (str), the name of the kernel, which is "bartlett" if it is not
            provided, ``bandwidth`` (float), and ``optimal_bw`` (bool). The names of
            the kernels are not case sensitive. See the parameters of
            :class:`~linearmodels.system.model.IVSystemGMM`.

        Returns
        -------
        IVSystemGMM
            Model instance.

        Raises
        ------
        UnknownOptionError
            If ``weight_type`` is not supported or ``kernel`` is not a supported
            kernel. It is both a ``KeyError`` and a ``ValueError``.

        Notes
        -----
        Models can be specified in one of two ways. The first uses curly
        braces to encapsulate equations.  The second uses a dictionary
        where each key is an equation name.

        Examples
        --------
        The simplest format uses standard formulas for each equation
        in a dictionary.  Best practice is to use an Ordered Dictionary

        >>> import pandas as pd
        >>> import numpy as np
        >>> cols = ["y1", "x1_1", "x1_2", "z1", "y2", "x2_1", "x2_2", "z2"]
        >>> data = pd.DataFrame(np.random.randn(500, 8), columns=cols)
        >>> from linearmodels.system import IVSystemGMM
        >>> formula = {"eq1": "y1 ~ 1 + x1_1 + [x1_2 ~ z1]",
        ...            "eq2": "y2 ~ 1 + x2_1 + [x2_2 ~ z2]"}
        >>> mod = IVSystemGMM.from_formula(formula, data)

        The second format uses curly braces {} to surround distinct equations

        >>> formula = "{y1 ~ 1 + x1_1 + [x1_2 ~ z1]} {y2 ~ 1 + x2_1 + [x2_2 ~ z2]}"
        >>> mod = IVSystemGMM.from_formula(formula, data)

        It is also possible to include equation labels when using curly braces

        >>> formula = "{eq1: y1 ~ x1_1 + [x1_2 ~ z1]} {eq2: y2 ~ 1 + [x2_2 ~ z2]}"
        >>> mod = IVSystemGMM.from_formula(formula, data)
        """
        context = capture_context(1)
        parser = SystemFormulaParser(formula, data, weights, context=context)
        eqns = parser.data
        mod = cls(eqns, sigma=None, weight_type=weight_type, **weight_config)
        mod.formula = formula
        return mod

    def _j_statistic(
        self,
        params: linearmodels.typing.data.Float64Array,
        weight_mat: linearmodels.typing.data.Float64Array,
    ) -> WaldTestStatistic:
        """
        J stat and test.

        Parameters
        ----------
        params : ndarray
            Estimated model parameters.
        weight_mat : ndarray
            Weighting matrix used in estimation of the parameters.

        Returns
        -------
        WaldTestStatistic
            Test statistic.

        Notes
        -----
        Assumes that the efficient weighting matrix has been used.  Using
        other weighting matrices will not produce the correct test.
        """
        y, x, z = self._wy, self._wx, self._wz
        k = len(x)
        ze_lst = []
        idx = 0
        for i in range(k):
            kx = x[i].shape[1]
            beta = params[idx : idx + kx]
            eps = y[i] - x[i] @ beta
            ze_lst.append(z[i] * eps)
            idx += kx
        ze = np.concatenate(ze_lst, 1)
        g_bar = ze.mean(0)
        nobs = x[0].shape[0]
        stat = float(nobs * g_bar.T @ np.linalg.inv(weight_mat) @ g_bar.T)
        null = "Expected moment conditions are equal to 0"
        ninstr = sum(a.shape[1] for a in z)
        nvar = sum(a.shape[1] for a in x)
        ncons = 0 if self.constraints is None else self.constraints.r.shape[0]

        return WaldTestStatistic(stat, null, ninstr - (nvar - ncons))
