"""Panel data models and estimators."""

from __future__ import annotations

from linearmodels.compat.scipy import csc_array, diags_array

from collections.abc import Mapping
from typing import Any, NamedTuple, Union, cast
import warnings

from formulaic.formula import Formula
from formulaic.model_spec import NAAction
from formulaic.parser.algos.tokenize import tokenize
from formulaic.utils.context import capture_context
import numpy as np
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    factorize,
    get_dummies,
)
from pandas.api.types import is_numeric_dtype
from scipy.linalg import lstsq as sp_lstsq
from scipy.sparse.linalg import lsmr

from linearmodels.iv.covariance import KERNEL_LOOKUP
from linearmodels.panel.covariance import (
    ACCovariance,
    ClusteredCovariance,
    CovarianceManager,
    DriscollKraay,
    FamaMacBethCovariance,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    check_covariance_kwargs,
    setup_covariance_estimator,
)
from linearmodels.panel.data import PanelArrayLike, PanelData, PanelDataLike
from linearmodels.panel.results import (
    FamaMacBethResults,
    PanelEffectsResults,
    PanelResults,
    RandomEffectsResults,
)
from linearmodels.panel.utility import (
    AbsorbingEffectWarning,
    absorbing_warn_msg,
    check_absorbed,
    dummy_matrix,
    in_2core_graph,
    not_absorbed,
)
from linearmodels.shared.alignment import (
    check_row_alignment,
    differs,
    is_default_index,
    is_labelled,
    labels_1d,
    misaligned,
    same_labels,
)
from linearmodels.shared.exceptions import (
    InferenceUnavailableWarning,
    MemoryWarning,
    MissingValueWarning,
    SingletonWarning,
    lookup_option,
    lower_option,
    missing_warning,
    unknown_option_error,
)
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
    WaldTestStatistic,
)
from linearmodels.shared.inputs import as_array_like, as_float64
from linearmodels.shared.linalg import has_constant
from linearmodels.shared.rsquared import r_squared, total_ss_is_zero
from linearmodels.shared.typed_getters import get_panel_data_like
from linearmodels.shared.utility import AttrDict, ensure_unique_column, panel_to_frame
from linearmodels.shared.weights import check_weights, normalize_weights
from linearmodels.typing import BoolArray
import linearmodels.typing.data

CovarianceEstimator = Union[
    ACCovariance,
    ClusteredCovariance,
    DriscollKraay,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
]

CovarianceEstimatorType = Union[
    type[ACCovariance],
    type[ClusteredCovariance],
    type[DriscollKraay],
    type[HeteroskedasticCovariance],
    type[HomoskedasticCovariance],
]


def _lstsq(
    x: linearmodels.typing.data.Float64Array,
    y: linearmodels.typing.data.Float64Array,
    rcond: float | None = None,
) -> tuple[
    linearmodels.typing.data.Float64Array,
    linearmodels.typing.data.Float64Array,
    int,
    linearmodels.typing.data.Float64Array,
]:
    if x.shape[1] == 0 or (y.ndim == 2 and y.shape[1] == 0):
        # There are no regressors, or no variables to regress
        empty = np.empty(0)
        return np.zeros((x.shape[1], *y.shape[1:])), empty, 0, empty
    if rcond is None:
        eps = np.finfo(np.float64).eps
        cond = float(max(x.shape) * eps)
    else:
        cond = rcond
    return sp_lstsq(x, y, cond=cond, lapack_driver="gelsy")


def panel_structure_stats(
    ids: linearmodels.typing.data.AnyIntArray, name: str
) -> Series:
    """
    Compute statistics of the number of observations in each group.

    Parameters
    ----------
    ids : ndarray
        The non-negative integer id of the group of each observation.
    name : str
        The name of the output, which is also used in the error message.

    Returns
    -------
    Series
        The mean, median, maximum, minimum and number of groups, named
        ``"mean"``, ``"median"``, ``"max"``, ``"min"`` and ``"total"``.

    Raises
    ------
    ValueError
        If an id is negative, which marks a missing value in the index of the
        panel.
    """
    ids = np.asarray(ids)
    if np.any(ids < 0):
        raise ValueError(
            f"{name} cannot be computed: ids must be non-negative integers. "
            "Negative ids indicate missing values (NaN) in the panel index."
        )
    bc = np.bincount(ids)
    bc = bc[bc > 0]
    index = ["mean", "median", "max", "min", "total"]
    out = [bc.mean(), np.median(bc), bc.max(), bc.min(), bc.shape[0]]
    return Series(out, index=index, name=name)


class FInfo(NamedTuple):  # numpydoc ignore=PR02
    """
    Information that the robust F-statistic of a model needs.

    Parameters
    ----------
    sel : ndarray
        Boolean array that selects the parameters that are tested, which are all
        of the parameters except for the constant.
    name : str
        The name of the test.
    invalid_test_stat : InvalidTestStatistic or None
        The test statistic to return if the test cannot be computed, or None.
    is_invalid : bool
        Flag indicating that the test cannot be computed.
    """

    sel: linearmodels.typing.data.BoolArray
    name: str
    invalid_test_stat: InvalidTestStatistic | None
    is_invalid: bool


# The model fits exactly, e.g., the dependent variable is all zero
_SINGULAR_COVARIANCE = "The covariance of the parameters is singular"


def _hotelling_f(
    params: Series,
    cov: DataFrame,
    debiased: bool,
    nperiod: int,
    f_info: FInfo,
) -> InvalidTestStatistic | WaldTestStatistic:
    """
    Robust Wald test of FamaMacBeth, with Hotelling's T-squared reference

    The parameters are the average of T estimates, and T is the argument nperiod.
    If debiased, the Wald statistic W is multiplied by (T - df) / (df * (T - 1))
    and compared with an F(df, T - df) distribution. This is Hotelling's
    T-squared test, which is exact when the estimates are independent and normal
    and the covariance is the covariance of their average. If not debiased the
    statistic is W, and the distribution is chi2(df). The covariance of df
    parameters that are averages of T estimates has rank at most T - 1, so that
    the statistic is invalid unless T is larger than df.
    """
    if f_info.is_invalid:
        assert f_info.invalid_test_stat is not None
        return f_info.invalid_test_stat
    sel = f_info.sel
    name = f_info.name
    df = int(sel.sum())
    null = "All parameters ex. constant are zero"
    if nperiod <= df:
        reason = (
            f"The test needs more periods than tested parameters. The {df} tested "
            f"parameters are the averages of the estimates of {nperiod} periods, "
            f"so that their covariance has rank at most {nperiod - 1}."
        )
        return InvalidTestStatistic(reason, name=name)
    test_params = np.asarray(params)[sel]
    test_cov = np.asarray(cov)[sel][:, sel]
    try:
        inv_test_cov = np.linalg.inv(test_cov)
    except np.linalg.LinAlgError:
        return InvalidTestStatistic(_SINGULAR_COVARIANCE, name=name)
    wald_stat = float(test_params.T @ inv_test_cov @ test_params)
    if not debiased:
        return WaldTestStatistic(wald_stat, null, df, name=name)
    scale = (nperiod - df) / (df * (nperiod - 1))
    return WaldTestStatistic(scale * wald_stat, null, df, nperiod - df, name=name)


def _deferred_f(
    params: Series,
    cov: DataFrame,
    debiased: bool,
    df_resid: int,
    f_info: FInfo,
) -> InvalidTestStatistic | WaldTestStatistic:
    if f_info.is_invalid:
        assert f_info.invalid_test_stat is not None
        return f_info.invalid_test_stat
    sel = f_info.sel
    name = f_info.name
    test_params = np.asarray(params)[sel]
    test_cov = np.asarray(cov)[sel][:, sel]
    try:
        inv_test_cov = np.linalg.inv(test_cov)
    except np.linalg.LinAlgError:
        return InvalidTestStatistic(_SINGULAR_COVARIANCE, name=name)
    test_stat = test_params.T @ inv_test_cov @ test_params
    test_stat = float(test_stat)
    df = int(sel.sum())
    null = "All parameters ex. constant are zero"

    if debiased:
        wald = WaldTestStatistic(test_stat / df, null, df, df_resid, name=name)
    else:
        wald = WaldTestStatistic(test_stat, null, df, name=name)
    return wald


def _panel_data(value: Any, name: str, var_name: str, **kwargs: Any) -> PanelData:
    """
    PanelData of an input of a model.

    Parameters
    ----------
    value : object
        The input.
    name : str
        The name of the parameter, which is added to the messages of the errors.
    var_name : str
        The name to use for the variables of NumPy inputs.
    **kwargs
        Keyword arguments for PanelData.

    Returns
    -------
    PanelData
        The panel data.
    """
    try:
        return PanelData(value, var_name, **kwargs)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: {exc}") from exc


class PanelFormulaParser:
    """
    Parse formulas for OLS and IV models.

    Parameters
    ----------
    formula : str
        String formula object.
    data : DataFrame
        Frame containing values for variables used in formula.
    eval_env : int, optional
        Stack depth to use when evaluating formulas. The default is 2. It is
        not used if `context` is provided.
    context : Mapping of str to object, optional
        The variables that the formula can use, by name. If not provided, it is
        captured at the stack depth `eval_env`.

    Notes
    -----
    The general structure of a formula is `dep ~ exog`.
    """

    def __init__(
        self,
        formula: str,
        data: PanelDataLike,
        eval_env: int = 2,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._formula = formula
        self._data = _panel_data(data, "data", "x", convert_dummies=False, copy=False)
        self._na_action = NAAction("ignore")
        self._eval_env = eval_env
        if not context:
            self._context = capture_context(eval_env)
        else:
            self._context = context
        self._dependent = self._exog = None
        self._parse()

    def _parse(self) -> None:
        parts = self._formula.split("~")
        parts[1] = " 0 + " + parts[1]
        cln_formula = "~".join(parts)
        rm_list = []
        effects = {"EntityEffects": False, "FixedEffects": False, "TimeEffects": False}
        tokens = tokenize(cln_formula)
        for token in tokens:
            _token = str(token)
            if _token in effects:
                effects[_token] = True
                rm_list.append(_token)
        if effects["EntityEffects"] and effects["FixedEffects"]:
            raise ValueError("Cannot use both FixedEffects and EntityEffects")
        self._entity_effect = effects["EntityEffects"] or effects["FixedEffects"]
        self._time_effect = effects["TimeEffects"]
        for effect, effect_val in effects.items():
            if effect_val:
                loc = cln_formula.find(effect)
                start = cln_formula.rfind("+", 0, loc)
                end = loc + len(effect)
                cln_formula = cln_formula[:start] + cln_formula[end:]
        self._lhs, self._rhs = ["0 + " + s.strip() for s in cln_formula.split("~")]

    @property
    def entity_effect(self) -> bool:
        """Formula contains entity effect."""
        return self._entity_effect

    @property
    def time_effect(self) -> bool:
        """Formula contains time effect."""
        return self._time_effect

    @property
    def eval_env(self) -> int:
        """Set or get the eval env depth."""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value: int) -> None:  # numpydoc ignore=GL08
        self._context = capture_context(value)
        self._eval_env = value

    @property
    def context(self) -> Mapping[str, Any]:
        """Get the context used in the parser."""
        return self._context

    @context.setter
    def context(self, value: Mapping[str, Any]) -> None:
        """
        Set the context used in the parser.

        Parameters
        ----------
        value : Mapping of str to object
            The variables that the formula can use, by name.
        """
        self._context = value

    @property
    def data(self) -> tuple[DataFrame, DataFrame]:
        """Return a tuple containing the dependent and exog."""
        out = self.dependent, self.exog
        return out

    @property
    def dependent(self) -> DataFrame:
        """DataFrame containing the dependent variable."""
        return DataFrame(
            Formula(self._lhs).get_model_matrix(
                self._data.dataframe,
                context=self._context,
                na_action=self._na_action,
            )
        )

    @property
    def exog(self) -> DataFrame:
        """DataFrame containing the exogenous variables."""
        return DataFrame(
            Formula(self._rhs).get_model_matrix(
                self._data.dataframe,
                context=self._context,
                na_action=self._na_action,
            )
        )


class AmbiguityError(Exception):
    """
    Error raised when the shape of the weights could mean different things.

    It is raised when the numbers of time periods and of entities are equal and
    1-d weights have that many values, so that the weights could have a value
    for each time period or for each entity.
    """


__all__ = [
    "AmbiguityError",
    "BetweenOLS",
    "FamaMacBeth",
    "FirstDifferenceOLS",
    "PanelOLS",
    "PooledOLS",
    "RandomEffects",
]


# Likely
# TODO: Formal test of other outputs
# Future
# TODO: Bootstrap covariance
# TODO: Possibly add AIC/BIC
# TODO: ML Estimation of RE model
# TODO: Defer estimation of 3 R2 values -- slow

EXOG_PREDICT_MSG = """\
exog does not have the correct number of columns. Saw {x_shape}, expected
{params_shape}. This can happen since exog is converted to a PanelData object
before computing the fitted value. The best practice is to pass a DataFrame
with a 2-level MultiIndex containing the entity- and time-ids."""


def _is_labelled(value: object) -> bool:
    """Whether the rows of a panel input are identified by labels.

    The columns of a DataFrame that has a default index also carry labels, since
    they identify the entities when the frame is time by entity. See
    :func:`linearmodels.shared.alignment.is_labelled`.
    """
    return is_labelled(value, columns=True)


def _is_grid(value: object) -> bool:
    """Whether a panel input has a value for each time period and entity.

    Such an input is a grid, with the time periods as its rows and the entities as
    its columns: a NumPy array with 2 dimensions (time by entity) or 3 dimensions
    (variable by time by entity), a DataFrame that does not have a MultiIndex, or
    a DataArray. A 1-d array, a Series or a DataFrame with a MultiIndex, and a
    PanelData, have a value for each row of dependent instead.
    """
    if isinstance(value, np.ndarray):
        return value.ndim != 1
    if isinstance(value, DataFrame):
        return not isinstance(value.index, MultiIndex)
    return not isinstance(value, (Series, PanelData))


def _grid_axis(labels: Index, groups: Index) -> Index:
    """The distinct labels of one axis of a grid, in the order of the grid.

    Parameters
    ----------
    labels : Index
        The label of this axis for each row of a panel, e.g., the time periods.
    groups : Index
        The label of the other axis for each row, e.g., the entities.

    Returns
    -------
    Index
        The distinct labels in the order of the grid. If the rows of every group
        have the labels in ascending order, it is the ascending order of the
        labels, and if they are in descending order it is the descending order.
        Otherwise it is the order in which the labels first appear in the rows.

    Notes
    -----
    The order of first appearance is the order of the labels if the rows are sorted
    and the panel is balanced, whether the rows are sorted by group and then label
    or by label and then group. It is not if the panel is not balanced and the first
    groups do not have the first labels, e.g., the first entity enters late and the
    rows are sorted by entity. The sorted order is used then. If the panel is
    balanced and the rows of every group are in order of the labels, they first
    appear in the same order, so that the rows are not examined.
    """
    appearance = labels.unique()
    if len(labels) == len(appearance) * len(groups.unique()):
        return appearance
    try:
        codes, ordered = factorize(labels, sort=True)
    except TypeError:
        # The labels cannot be ordered
        return appearance
    if len(ordered) != len(appearance):
        return appearance
    group_codes = np.asarray(factorize(groups)[0])
    rows = np.argsort(group_codes, kind="stable")
    group_codes, codes = group_codes[rows], np.asarray(codes)[rows]
    # The steps between the rows of a group, which are in the order of the rows
    steps = np.diff(codes)[group_codes[1:] == group_codes[:-1]]
    if np.all(steps > 0):
        return ordered
    if np.all(steps < 0):
        return ordered[::-1]
    return appearance


def _grid_axes(index: Index) -> tuple[Index, Index]:
    """The entities and time periods in the order of a grid placed on the rows.

    Parameters
    ----------
    index : Index
        The rows of a panel, a MultiIndex with the entities in the first level and
        the time periods in the second.

    Returns
    -------
    entities : Index
        The distinct entities, which are the columns of the grid.
    times : Index
        The distinct time periods, which are the rows of the grid.
    """
    entities = index.get_level_values(0)
    times = index.get_level_values(1)
    return _grid_axis(entities, times), _grid_axis(times, entities)


class _PanelModelBase:
    r"""
    Base class for all panel models

    Parameters
    ----------
    dependent : PanelData, DataFrame, Series, ndarray or DataArray
        Dependent (left-hand-side) variable (time by entity). A DataFrame or
        Series with a 2-level MultiIndex (entity, then time), a DataFrame with
        time as the index and entities as the columns, a 2-d (time by entity)
        NumPy array or DataArray, or a PanelData. A list, a tuple or any other
        container raises a ``TypeError`` that names ``dependent``, since the
        entities and time periods need labels or a layout.
    exog : PanelData, DataFrame, Series, ndarray or DataArray
        Exogenous or right-hand-side variables (variable by time by entity). The
        containers are those of ``dependent``, where a DataFrame has a column for
        each variable and an array or DataArray with several variables has 3
        dimensions (variable by time by entity). A list, a tuple or any other
        container raises a ``TypeError`` that names ``exog``.
    weights : array_like, optional
        Weights to use in estimation. Assumes the variance of the residual is
        proportional to the inverse of the weight, so that the residual times
        the square root of the weight is homoskedastic.
        Weights can be given for each observation, for each time period,
        for each entity or for each combination of time period and entity.
        The weights must be strictly positive and finite. They are rescaled to have a
        mean of 1. A NaN weight is missing, and its observation is dropped.
        Weights with labels must use the same labels, in the same order,
        as dependent or, if dependent has no labels, as exog. See
        :ref:`panel-input-alignment`.
        Dependent need not be balanced when the weights are given for each time
        period, entity or combination: the weights of the combinations that
        dependent does not have are not used.
        A 1-d array has a value for each row of dependent, in the order of the
        rows, if it has as many values as dependent has rows. It has a value for
        each time period if it has as many values as there are time periods, and
        a value for each entity if it has as many as there are entities. Weights
        with labels (a pandas object with an index that is not the default index)
        are read by their labels, so that a Series indexed by the time periods has
        a value for each time period, and one indexed by the entities has a value
        for each entity. Weights without labels (an array, a list, a tuple, or a
        Series or a DataFrame with one column that has the default index) are read
        by their number of values. If dependent is not balanced and has as many
        rows as time periods or entities, they have a value for each row. An
        ``AmbiguityError`` is raised if the numbers of time periods and entities
        are equal and the weights have that many values, unless they have no
        labels and a value for each row. Any other number of values raises a
        ``ValueError``.
    check_rank : bool, optional
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank. The default is True.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> None:
        self.dependent = _panel_data(dependent, "dependent", "Dep")
        self.exog = _panel_data(exog, "exog", "Exog")
        # The inputs with labels use the labels of dependent. If dependent has no
        # labels, they use the labels of the first of them that has labels
        self._labels: Index | None = None
        self._labels_name = "dependent"
        self._check_labels("dependent", self.dependent.index, _is_labelled(dependent))
        self._check_labels("exog", self.exog.index, _is_labelled(exog))
        self._original_shape = self.dependent.shape
        # The rows of dependent, before the observations with missing values are
        # dropped. The inputs that are given for each time period and entity are
        # placed on these rows
        self._original_index = self.dependent.index.copy()
        self._constant = False
        self._formula: str | None = None
        self._is_weighted = True
        self._name = self.__class__.__name__
        self.weights = self._adapt_weights(weights)
        self._not_null: BoolArray = np.ones(
            self.dependent.values2d.shape[0], dtype=bool
        )
        self._cov_estimators = CovarianceManager(
            self.__class__.__name__,
            HomoskedasticCovariance,
            HeteroskedasticCovariance,
            ClusteredCovariance,
            DriscollKraay,
            ACCovariance,
        )
        self._constant_index: int | None = None
        self._check_rank = bool(check_rank)
        self._validate_data()
        self._singleton_index: linearmodels.typing.data.BoolArray | None = None
        self._check_regressors()

    def _check_regressors(self) -> None:
        """
        Raise a ValueError if exog has no columns

        Raises
        ------
        ValueError
            If exog has no columns. Only PanelOLS with effects can estimate a model
            without regressors.
        """
        if self.exog.nvar == 0:
            raise ValueError(
                f"{self.__class__.__name__} needs at least one regressor, but exog "
                "has no columns. Only PanelOLS can estimate a model without "
                "regressors, and it needs entity, time or other effects."
            )

    def __str__(self) -> str:
        out = "{name} \nNum exog: {num_exog}, Constant: {has_constant}"
        return out.format(
            name=self.__class__.__name__,
            num_exog=self.exog.dataframe.shape[1],
            has_constant=self.has_constant,
        )

    def __repr__(self) -> str:
        return self.__str__() + "\nid: " + str(hex(id(self)))

    def _check_labels(self, name: str, labels: Index, labelled: bool) -> None:
        """
        Check the labels of an input against the labels of the other inputs

        Parameters
        ----------
        name : str
            The name of the input, which is used in the message.
        labels : Index
            The labels of the rows of the input.
        labelled : bool
            Whether the input carries labels. If not, it is paired with the
            other inputs by position and there is nothing to check.

        Notes
        -----
        The labels of the first input with labels are saved, so that the inputs
        that follow it have to use them.
        """
        if not labelled:
            return
        if self._labels is None:
            if isinstance(labels, MultiIndex):
                self._labels, self._labels_name = labels, name
            return
        check_row_alignment(name, labels, self._labels, self._labels_name)

    def _panel_rows(self, value: Any, name: str, var_name: str) -> PanelData:
        """
        PanelData of an input that has the shape of the data, e.g., clusters

        Parameters
        ----------
        value : array_like
            The input. A 1-d array has a value for each row of dependent, in the
            order of the rows of dependent, since it only has one variable.
            Anything else follows the rules of PanelData, after lists and tuples
            are converted to arrays.
        name : str
            The name of the parameter, which is added to the messages of the
            errors.
        var_name : str
            The name to use for the variable when it is a NumPy array.

        Returns
        -------
        PanelData
            The input as panel data, without the conversion of categories to
            dummy variables.
        """
        value = as_array_like(value, name)
        if isinstance(value, np.ndarray) and value.ndim == 1:
            nrows = len(self._original_index)
            if value.shape[0] != nrows:
                raise ValueError(
                    f"{name} is 1-d and has {value.shape[0]} values, but dependent "
                    f"has {nrows} rows. A 1-d array must have a value for each row "
                    "of dependent, in the order of the rows of dependent."
                )
            value = DataFrame({var_name: value}, index=self._original_index)
        return _panel_data(value, name, var_name, convert_dummies=False)

    def reformat_clusters(
        self, clusters: linearmodels.typing.data.IntArray | PanelArrayLike
    ) -> PanelData:
        """
        Reformat cluster variables.

        Parameters
        ----------
        clusters : array_like
            Values to use for variance clustering. A 1-d array has one value
            for each row of dependent, in the order of the rows of dependent.
            A 2-d array (time by entity), a 3-d array (variable by time by
            entity), a DataFrame with the time periods as its index and the
            entities as its columns and a DataArray have a value for each
            time period and entity. They are placed on the rows of dependent,
            which need not be balanced: the values of the time periods and
            entities that dependent does not have are not used.

        Returns
        -------
        PanelData
            Original data with matching axis and observation dropped where
            missing in the model data.

        Raises
        ------
        ValueError
            If clusters does not have the numbers of entities and time periods of
            dependent, or has labels that are not those of dependent in the same
            order. A value for each time period and entity must have the time
            periods and entities in the order that the Notes describe. Also if
            clusters has a missing value for an observation that the model uses,
            i.e., one that is not dropped because dependent, exog, weights or
            other_effects has a missing value.

        Notes
        -----
        This is exposed for testing and is not normally needed for estimation

        The time periods and entities of a value for each time period and entity
        are matched to those of dependent in the order in which they first appear
        in the rows of dependent. If dependent is not balanced and the first entity
        does not have the first periods, the periods of a panel that is sorted by
        entity first appear in an order that is not their order in time. The time
        periods are in order of time then, if the rows of every entity are in order
        of time (in either direction), and the entities are in the order of their
        labels if the rows of every time period are in order of the entities. A
        pandas object or DataArray with labels must have the time periods and
        entities in this order.

        .. versionchanged:: 7.1

           A value for each time period and entity can be used if dependent is
           not balanced. It raised ``IndexError: Boolean index has wrong length``.
           It is placed on the rows of dependent by time period and entity.
           Earlier versions paired it by position with the rows of dependent, which
           is right only if the rows are sorted by entity and then time. If they
           were sorted by time and then entity, an array was attached to the wrong
           observations without an error, and a pandas object with labels was
           rejected.
        """
        clusters = as_array_like(clusters, "clusters")
        labelled = _is_labelled(clusters)
        clusters_pd = self._panel_rows(clusters, "clusters", "cov.cluster")
        if clusters_pd.shape[1:] != self._original_shape[1:]:
            raise ValueError(
                "clusters must have the same number of entities "
                "and time periods as the model data."
            )
        if _is_grid(clusters):
            # Compared with the labels of the model if it has any. Unlike the other
            # inputs, the clusters never set the labels, as they are given to fit
            clusters_pd = self._grid_to_rows(
                clusters_pd, labelled and self._labels is not None, "clusters"
            )
        elif labelled and self._labels is not None:
            check_row_alignment(
                "clusters", clusters_pd.index, self._labels, self._labels_name
            )
        frame = clusters_pd.dataframe
        changed = False
        if frame.shape[0] == len(self._original_index) and not frame.index.equals(
            self._original_index
        ):
            # The values are paired with the rows of dependent by position, and so
            # they have the labels of these rows
            frame = frame.set_axis(self._original_index, axis=0)
            changed = True
        # This comes before text and categories are replaced by codes, which give a
        # missing value the code -1 and so a cluster of its own. The clusters of the
        # observations that the model drops are not used, and can be missing
        if frame.loc[self.not_null].isnull().to_numpy().any():
            raise ValueError("clusters cannot contain missing values.")
        categories = [col for col in frame if not is_numeric_dtype(frame[col].dtype)]
        if categories:
            # Text and categories identify the clusters, but they are not numbers
            frame = frame.copy()
            for col in categories:
                frame[col] = Categorical(frame[col]).codes.astype(np.int64)
            changed = True
        if changed:
            clusters_pd = PanelData(frame, convert_dummies=False)
        clusters_pd.drop(~self.not_null)
        return clusters_pd.copy()

    def _info(self) -> tuple[Series, Series, DataFrame | None]:
        """Information about panel structure"""

        entity_info = panel_structure_stats(
            self.dependent.entity_ids.squeeze(), "Observations per entity"
        )
        time_info = panel_structure_stats(
            self.dependent.time_ids.squeeze(), "Observations per time period"
        )
        other_info = None

        return entity_info, time_info, other_info

    def _adapt_weights(self, weights: PanelArrayLike | None) -> PanelData:
        """
        Check the weights and place them on the rows of dependent

        Parameters
        ----------
        weights : array_like or None
            The weights. None gives every observation the same weight.

        Returns
        -------
        PanelData
            The weights of the rows of dependent, before the observations with
            missing values are dropped.

        Raises
        ------
        AmbiguityError
            If the number of time periods is the number of entities and 1-d weights
            have this number of values, so that they could be a value for each time
            period or for each entity.
        ValueError
            If the weights do not have a supported shape, or have labels that are
            not those of the other inputs.

        Notes
        -----
        The shape of the weights decides how they are read. Let nobs be the number
        of time periods, nentity the number of entities and nrows the number of rows
        of dependent. nrows is nobs * nentity if dependent is balanced and less if it
        is not. The weights have

        * a value for each combination of a time period and an entity if they are a
          3-d array, a PanelData or have the shape (nobs, nentity);
        * a value for each row if they are 1-d with nrows values. A pandas object
          with a MultiIndex is paired by label, and anything else by position
          with the rows of dependent;
        * a value for each time period if they are 1-d with nobs values;
        * a value for each entity if they are 1-d with nentity values.

        If nrows is also nobs or nentity, labels decide. Weights with labels, a
        pandas object with an index that is not the default index, are read by their
        labels: a Series indexed by the time periods is a value for each time period
        and one indexed by the entities is a value for each entity. Weights without
        labels, which are an array, a list, a tuple, and a Series or a DataFrame with
        one column that has the default index, are read by their number of values. A
        DataFrame with one column is read as an array with a column is if it has the
        default index, and as a Series with the same index if it has any other. If dependent
        is not balanced, this happens if every time period or entity has a single
        row, and they are a value for each row. If it is balanced, this happens if
        there is a single entity or time period, and they are a value for each time
        period or entity.

        An ``AmbiguityError`` is raised if nobs and nentity are equal and 1-d weights
        have that many values, since they could be either, and labels do not decide
        it. The exception is weights without labels that are a value for each row of
        a dependent that is not balanced. Any other number of values raises a
        ``ValueError`` that gives the accepted numbers.
        """
        if weights is None:
            self._is_weighted = False
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = 1
            # TODO: Remove once pandas typing fixed
            frame.columns = Index(["weight"])
            return PanelData(frame)

        # Lists and tuples are converted to arrays, so that ndim and shape exist
        weights = as_array_like(weights, "weights")
        if isinstance(weights, np.ndarray):
            weights = as_float64(weights, "weights")
        nobs, nentity = self.exog.nobs, self.exog.nentity
        nrows = self.dependent.dataframe.shape[0]
        entities, times = _grid_axes(self.dependent.index)
        labelled = _is_labelled(weights)
        # Weights for each time period or entity have to use the time periods or
        # entities of the inputs with labels, if there are any
        reference = self._labels
        if reference is not None:
            ref_entities, ref_times = _grid_axes(reference)

        if weights.ndim == 3 or weights.shape == (nobs, nentity):
            panel = _panel_data(weights, "weights", "x")
            if isinstance(weights, PanelData):
                self._check_labels("weights", panel.index, labelled)
                return panel
            return self._grid_to_rows(panel, labelled)

        if isinstance(weights, np.ndarray):
            weights = cast("linearmodels.typing.data.Float64Array", np.squeeze(weights))
        elif (
            isinstance(weights, DataFrame)
            and weights.shape[1] == 1
            and is_default_index(weights.index)
        ):
            # A column of values with a default index has no labels. It is read as the
            # array with a column is, whatever the name of the column is
            column = as_float64(np.asarray(weights), "weights")
            weights = cast("linearmodels.typing.data.Float64Array", column[:, 0])
            labelled = False
        elif (
            isinstance(weights, DataFrame)
            and weights.shape[1] == 1
            and not isinstance(weights.index, MultiIndex)
        ):
            # The labels decide, as they do for a Series with the same index
            weights = weights.iloc[:, 0]
        unbalanced = nrows != nobs * nentity
        # The labels of a pandas object with a MultiIndex identify the rows
        row_labels = (
            isinstance(weights, (Series, DataFrame))
            and isinstance(weights.index, MultiIndex)
            and weights.shape[0] == nrows
        )
        # 1-d weights with a value for each row of dependent. If dependent is not
        # balanced it has fewer rows than combinations of a time period and an entity
        # (nobs * nentity), so that the weights cannot be a value for each combination.
        # Weights without labels are a value for each row also if the number of rows is
        # nobs or nentity, which happens if every time period or entity has a single
        # row. Weights with labels are read by their labels in that case, so that a
        # Series indexed by the time periods is a value for each time period, and one
        # indexed by the entities a value for each entity. If dependent is balanced, a
        # single entity or time period makes the number of rows nobs or nentity, and the
        # weights are read as before, as a value for each time period or entity
        by_row = (
            weights.ndim == 1
            and weights.shape[0] == nrows
            and (
                weights.shape[0] not in (nobs, nentity) or (unbalanced and not labelled)
            )
        )
        if (
            weights.shape[0] == nobs
            and nobs == nentity
            and not by_row
            and not (row_labels and unbalanced)
        ):
            raise AmbiguityError(
                "Unable to distinguish nobs from nentity since they are "
                "equal. You must use a 2-d array to avoid ambiguity."
            )
        if row_labels:
            panel = PanelData(DataFrame(weights))
            self._check_labels("weights", panel.index, labelled)
            return panel
        elif by_row:
            return self._weights_by_row(weights, labelled)
        elif weights.shape[0] == nobs and weights.ndim == 1:
            if (
                labelled
                and reference is not None
                and differs(labels_1d(weights), ref_times)
            ):
                raise misaligned("weights", "The time index", self._labels_name)
            weights_arr = np.asarray(weights)[:, None]
            weights_arr = weights_arr @ np.ones((1, nentity))
            grid = DataFrame(weights_arr, index=times, columns=entities)
        elif weights.shape[0] == nentity and weights.ndim == 1:
            if (
                labelled
                and reference is not None
                and differs(labels_1d(weights), ref_entities)
            ):
                raise misaligned("weights", "The entity index", self._labels_name)
            weights_arr = np.asarray(weights)[None, :]
            weights_arr = np.ones((nobs, 1)) @ weights_arr
            grid = DataFrame(weights_arr, index=times, columns=entities)
        else:
            ncells = nobs * nentity
            msg = (
                "Weights do not have a supported shape. The weights have shape "
                f"{weights.shape}. 1-d weights must have {nrows} values (one for "
                f"each row of dependent), {nobs} values (one for each time period) "
                f"or {nentity} values (one for each entity). 2-d weights must have "
                f"{nobs} rows (time periods) and {nentity} columns (entities)."
            )
            if weights.ndim == 1 and weights.shape[0] == ncells and ncells != nrows:
                msg += (
                    f" dependent has {nrows} rows and not {ncells}, the number of "
                    "combinations of a time period and an entity, since it is not "
                    "balanced."
                )
            raise ValueError(msg)
        return self._grid_to_rows(PanelData(grid), False)

    def _weights_by_row(self, weights: Any, labelled: bool) -> PanelData:
        """
        Place weights that have a value for each row on the rows of dependent

        Parameters
        ----------
        weights : array_like
            1-d weights with as many values as dependent has rows. The values are
            paired with the rows of dependent by position, whatever the order of the
            rows is, as a 1-d array of clusters or other effects is.
        labelled : bool
            Whether the weights carry labels, which have to be those of the inputs
            with labels.

        Returns
        -------
        PanelData
            The weights, with the rows of dependent before the observations with
            missing values are dropped.
        """
        if labelled:
            self._check_labels("weights", labels_1d(weights), labelled)
        frame = self.dependent.dataframe.copy()
        # TODO: Fix this for pandas 3
        frame.iloc[:, :] = np.asarray(weights)[:, None]
        return PanelData(frame)

    def _grid_to_rows(
        self, grid: PanelData, labelled: bool, name: str = "weights"
    ) -> PanelData:
        """
        Place values defined on a time by entity grid on the rows of dependent

        Parameters
        ----------
        grid : PanelData
            Values for every entity and time period, stacked by entity.
        labelled : bool
            Whether the grid carries labels, which have to match the entities and
            time periods of the inputs with labels. If False, the position along
            each axis of the grid is paired with the entities and time periods
            of dependent in order of appearance, see ``_grid_axes``.
        name : str, optional
            The name of the input, which is used in the messages. The default is
            "weights".

        Returns
        -------
        PanelData
            Values with the rows of dependent, before the observations with
            missing values were dropped. If dependent is not balanced, the values
            of the combinations of a time period and an entity that it does not
            have are not used. If the grid has the wrong shape then grid is
            returned, and the caller reports the mismatch.
        """
        index = self._original_index
        grid_index = MultiIndex.from_product(_grid_axes(index))
        rows = grid.dataframe
        if len(rows) != len(grid_index):
            return grid
        if labelled and self._labels is None:
            self._check_labels(name, rows.index, labelled)
        elif labelled:
            assert self._labels is not None
            expected = MultiIndex.from_product(
                _grid_axes(self._labels),
            )
            if len(expected) == len(rows) and not same_labels(rows.index, expected):
                raise misaligned(
                    name,
                    "The time and entity index",
                    self._labels_name,
                )
        rows = rows.set_axis(grid_index, axis=0)
        if not grid_index.equals(index):
            rows = rows.reindex(index)
        # Text and categories, e.g., the clusters, are not converted to dummy variables
        return PanelData(rows, convert_dummies=False)

    def _check_exog_rank(self) -> int:
        x = cast("linearmodels.typing.data.Float64Array", self.exog.values2d)
        if not self._check_rank:
            # The shape of the PanelData is (variables, periods, entities)
            return x.shape[1]
        _, _, rank_of_x, _ = _lstsq(x, np.ones(x.shape[0]))
        if rank_of_x < x.shape[1]:
            raise ValueError(
                "exog does not have full column rank. If you wish to proceed with "
                "model estimation irrespective of the numerical accuracy of "
                "coefficient estimates, you can set check_rank=False."
            )
        return rank_of_x

    def _missing_in_other_inputs(self) -> BoolArray | None:
        """
        Observations that are missing in inputs other than the data of the model

        Returns
        -------
        ndarray or None
            A boolean array with a value for each row of dependent, which is True
            for the observations to drop since an input that the model validates,
            e.g., the other effects of PanelOLS, is missing. None if there are
            no such inputs.

        Notes
        -----
        The observations are dropped together with those that are missing in
        dependent, exog and weights, so that there is a single warning and the
        weights and the constant are determined from the observations that remain.
        """
        return None

    def _validate_data(self) -> None:
        """Check input shape and remove missing"""
        y = self._y = cast(
            "linearmodels.typing.data.Float64Array", self.dependent.values2d
        )
        x = self._x = cast("linearmodels.typing.data.Float64Array", self.exog.values2d)
        w = self._w = cast(
            "linearmodels.typing.data.Float64Array", self.weights.values2d
        )
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                "dependent and exog must have the same number of "
                "observations. The number of observations in dependent "
                f"is {y.shape[0]}, and the number of observations in exog "
                f"is {x.shape[0]}."
            )
        if y.shape[0] != w.shape[0]:
            raise ValueError(
                "weights must have the same number of observations as dependent."
            )

        # The rows that are missing in dependent and in all of the regressors are
        # dropped without a warning. If there are no regressors, the rows that are
        # missing in dependent only get the warning
        all_missing = np.any(np.isnan(y), axis=1) & (
            np.all(np.isnan(x), axis=1) & (x.shape[1] > 0)
        )
        missing = np.asarray(
            np.any(np.isnan(y), axis=1)
            | np.any(np.isnan(x), axis=1)
            | np.any(np.isnan(w), axis=1),
            dtype=bool,
        )
        missing_in_other_inputs = self._missing_in_other_inputs()
        if missing_in_other_inputs is not None:
            missing = missing | missing_in_other_inputs

        missing_warning(np.asarray(all_missing ^ missing), stacklevel=4)
        if np.any(missing):
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.weights.drop(missing)

            x = cast("linearmodels.typing.data.Float64Array", self.exog.values2d)
            self._not_null = cast("BoolArray", np.asarray(~missing))

        w_df = self.weights.dataframe
        w_values = np.asarray(w_df)
        check_weights(w_values)
        w_df = DataFrame(
            normalize_weights(w_values), index=w_df.index, columns=w_df.columns
        )
        self.weights = PanelData(w_df)
        rank_of_x = self._check_exog_rank()
        self._constant, self._constant_index = has_constant(x, rank_of_x)

    @property
    def formula(self) -> str | None:
        """Formula used to construct the model."""
        return self._formula

    @formula.setter
    def formula(self, value: str | None) -> None:  # numpydoc ignore=GL08
        self._formula = value

    @property
    def has_constant(self) -> bool:
        """Flag indicating the model has a constant or implicit constant."""
        return self._constant

    def _constant_loadings(self) -> linearmodels.typing.data.Float64Array:
        """
        Coefficients of the regressors that combine them into a constant

        Returns
        -------
        ndarray
            The (nvar, 1) coefficients a such that exog @ a is a column of ones.
            They exist if the model has a constant, which can be implicit.

        Notes
        -----
        The data are weighted or transformed before the model is estimated. The
        constant is transformed in the same way, and it is not a column of ones.
        Since the transformations are linear, it is the transformed regressors
        times these coefficients.
        """
        exog = self.exog.values2d
        ones = np.ones((exog.shape[0], 1))
        return _lstsq(exog, ones)[0]

    def _f_statistic(
        self,
        weps: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        df_resid: int,
    ) -> WaldTestStatistic | InvalidTestStatistic:
        """Compute model F-statistic"""
        weps_const = y
        num_df = x.shape[1]
        name = "Model F-statistic (homoskedastic)"
        if num_df == 0:
            return InvalidTestStatistic("Model contains no regressors", name=name)
        if df_resid <= 0:
            return InvalidTestStatistic(
                "The model has no residual degrees of freedom", name=name
            )
        if self.has_constant:
            if num_df == 1:
                return InvalidTestStatistic("Model contains only a constant", name=name)

            num_df -= 1
            # The restricted model only has the transformed constant
            const = x @ self._constant_loadings()
            mean = float(np.squeeze((const.T @ y) / (const.T @ const)))
            weps_const = cast("linearmodels.typing.data.Float64Array", y - const * mean)

        resid_ss = float(np.squeeze(weps.T @ weps))
        num = float(np.squeeze(weps_const.T @ weps_const - resid_ss))
        denom = resid_ss
        denom_df = df_resid
        stat = float((num / num_df) / (denom / denom_df)) if denom > 0.0 else 0.0
        return WaldTestStatistic(
            stat,
            null="All parameters ex. constant are zero",
            df=num_df,
            df_denom=denom_df,
            name=name,
        )

    def _f_statistic_robust(
        self,
        params: linearmodels.typing.data.Float64Array,
    ) -> FInfo:
        """Compute Wald test that all parameters are 0, ex. constant"""
        sel = np.ones(params.shape[0], dtype=bool)
        name = "Model F-statistic (robust)"

        if len(sel) == 0:
            return FInfo(
                sel,
                name,
                InvalidTestStatistic("Model contains no regressors", name=name),
                True,
            )
        if self.has_constant:
            if len(sel) == 1:
                return FInfo(
                    sel,
                    name,
                    InvalidTestStatistic("Model contains only a constant", name=name),
                    True,
                )
            assert isinstance(self._constant_index, int)
            sel[self._constant_index] = False

        return FInfo(sel, name, None, False)

    def _prepare_between(
        self,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """Prepare values for between estimation of R2"""
        weights = self.weights if self._is_weighted else None
        y = np.asarray(self.dependent.mean("entity", weights=weights))
        x = np.asarray(self.exog.mean("entity", weights=weights))
        # Weight transformation
        wcount, wmean = self.weights.count("entity"), self.weights.mean("entity")
        wsum = wcount * wmean
        w = np.asarray(wsum)
        w = w / w.mean()

        return y, x, w

    def _rsquared_corr(
        self, params: linearmodels.typing.data.Float64Array
    ) -> tuple[float, float, float]:
        """Correlation-based measures of R2"""

        def corr_squared(
            y: linearmodels.typing.data.Float64Array,
            xb: linearmodels.typing.data.Float64Array,
            y_ss: float,
        ) -> float:
            # Undefined if y does not vary. 0 if the prediction does not vary
            if total_ss_is_zero(float(((y - y.mean()) ** 2).sum()), y_ss, y.shape[0]):
                return np.nan
            return np.corrcoef(y.T, xb.T)[0, 1] ** 2 if xb.std() > 0 else 0.0

        # Overall
        y = self.dependent.values2d
        x = self.exog.values2d
        y_ss = float((y**2).sum())
        r2o = corr_squared(y, x @ params, y_ss)
        # Between
        y = np.asarray(self.dependent.mean("entity"))
        x = np.asarray(self.exog.mean("entity"))
        r2b = corr_squared(y, x @ params, float((y**2).sum()))

        # Within
        y = self.dependent.demean("entity", return_panel=False)
        x = self.exog.demean("entity", return_panel=False)
        r2w = corr_squared(y, x @ params, y_ss)

        return r2o, r2w, r2b

    def _rsquared(
        self, params: linearmodels.typing.data.Float64Array, reweight: bool = False
    ) -> tuple[float, float, float]:
        """Compute alternative measures of R2"""
        #############################################
        # R2 - Between
        #############################################
        y, x, w = self._prepare_between()
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(w)
        else:
            root_w = cast("linearmodels.typing.data.Float64Array", np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        e = y
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(np.squeeze(w.T @ (e**2)))
        y_ss = float(np.squeeze(w.T @ (y**2)))
        r2b = r_squared(residual_ss, total_ss, y_ss, y.shape[0])

        #############################################
        # R2 - Overall
        #############################################
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = cast("linearmodels.typing.data.Float64Array", np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        mu = (w * y).sum() / w.sum() if self.has_constant else 0
        we = wy - root_w * mu
        total_ss = float(np.squeeze(we.T @ we))
        y_ss = float(np.squeeze(wy.T @ wy))
        r2o = r_squared(residual_ss, total_ss, y_ss, y.shape[0])

        #############################################
        # R2 - Within
        #############################################
        weights = self.weights if self._is_weighted else None
        wy = cast(
            "linearmodels.typing.data.Float64Array",
            self.dependent.demean("entity", weights=weights, return_panel=False),
        )
        wx = cast(
            "linearmodels.typing.data.Float64Array",
            self.exog.demean("entity", weights=weights, return_panel=False),
        )
        assert isinstance(wy, np.ndarray)
        assert isinstance(wx, np.ndarray)
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        total_ss = float(np.squeeze(wy.T @ wy))
        r2w = r_squared(residual_ss, total_ss, y_ss, wy.shape[0])

        if self.has_constant and self.exog.nvar == 1:
            # A model with only a constant explains nothing, whatever the data
            r2b, r2o, r2w = (np.nan if np.isnan(r2) else 0.0 for r2 in (r2b, r2o, r2w))

        return r2o, r2w, r2b

    def _postestimation(
        self,
        params: linearmodels.typing.data.Float64Array,
        cov: CovarianceEstimator,
        debiased: bool,
        df_resid: int,
        weps: linearmodels.typing.data.Float64Array,
        y: linearmodels.typing.data.Float64Array,
        x: linearmodels.typing.data.Float64Array,
        weights: linearmodels.typing.data.Float64Array | None = None,
    ) -> AttrDict:
        """
        Common post-estimation values

        The log-likelihood is that of the Gaussian regression whose weighted
        residuals are ``weps`` = sqrt(w) * e, where the variance of the error of an
        observation is sigma2 / w. Its Jacobian term is 0.5 * sum(log(w)), where w
        is ``weights``, the weight of each of the observations of the regression.
        There are no weights, or unit weights, if ``weights`` is None.
        """
        f_info = self._f_statistic_robust(params)
        f_stat = self._f_statistic(weps, y, x, df_resid)
        r2o, r2w, r2b = self._rsquared(params)
        c2o, c2w, c2b = self._rsquared_corr(params)
        f_pooled = InapplicableTestStatistic(
            reason="Model has no effects", name="Pooled F-stat"
        )
        entity_info, time_info, other_info = self._info()
        nobs = weps.shape[0]
        sigma2 = float(np.squeeze(weps.T @ weps) / nobs)
        if sigma2 > 0.0:
            loglik: float = -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
            if weights is not None:
                loglik += 0.5 * float(np.sum(np.log(weights)))
        else:
            loglik = np.nan

        res = AttrDict(
            params=params,
            deferred_cov=cov.deferred_cov,
            f_info=f_info,
            f_stat=f_stat,
            debiased=debiased,
            name=self._name,
            var_names=self.exog.vars,
            r2w=r2w,
            r2b=r2b,
            r2=r2w,
            r2o=r2o,
            c2o=c2o,
            c2b=c2b,
            c2w=c2w,
            s2=cov.s2,
            model=self,
            cov_type=cov.name,
            index=self.dependent.index,
            entity_info=entity_info,
            time_info=time_info,
            other_info=other_info,
            f_pooled=f_pooled,
            loglik=loglik,
            not_null=self._not_null,
            original_index=self._original_index,
        )
        return res

    @property
    def not_null(self) -> linearmodels.typing.data.BoolArray:
        """Locations of non-missing observations."""
        return self._not_null

    def _setup_clusters(
        self,
        cov_config: Mapping[
            str,
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData,
        ],
    ) -> dict[
        str,
        bool | float | str | linearmodels.typing.data.IntArray | DataFrame | PanelData,
    ]:
        cov_config_upd = dict(cov_config)
        cluster_types = ("clusters", "cluster_entity", "cluster_time")
        common = set(cov_config.keys()).intersection(cluster_types)
        if not common:
            return cov_config_upd

        cov_config_upd = dict(cov_config.items())

        given = cov_config.get("clusters")
        if given is not None:
            # Lists and tuples are converted to arrays
            given = as_array_like(given, "clusters")
        clusters = get_panel_data_like({"clusters": given}, "clusters")
        clusters_frame: DataFrame | None = None
        if clusters is not None:
            formatted_clusters = self.reformat_clusters(clusters)
            for col in formatted_clusters.dataframe:
                cat = Categorical(formatted_clusters.dataframe[col])
                # TODO: Bug in pandas-stubs
                #  https://github.com/pandas-dev/pandas-stubs/issues/111
                formatted_clusters.dataframe[col] = cat.codes.astype(np.int64)
            clusters_frame = formatted_clusters.dataframe
            if self._singleton_index is not None:
                # The clusters have a value for each observation that is not missing.
                # The singletons are dropped from the data of the model, and so are
                # the clusters that are made from it below, which do not need this.
                # The copy is not a slice of the frame, which the columns that are
                # added below would be set on
                clusters_frame = clusters_frame.loc[~self._singleton_index].copy()

        cluster_entity = bool(cov_config_upd.pop("cluster_entity", False))
        if cluster_entity:
            group_ids_arr = self.dependent.entity_ids.squeeze()
            name = "cov.cluster.entity"
            group_ids: Series[int] = Series(
                group_ids_arr, index=self.dependent.index, name=name
            )
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)

        cluster_time = bool(cov_config_upd.pop("cluster_time", False))
        if cluster_time:
            group_ids_arr = self.dependent.time_ids.squeeze()
            name = "cov.cluster.time"
            group_ids = Series(group_ids_arr, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)

        if clusters_frame is not None:
            cov_config_upd["clusters"] = np.asarray(clusters_frame)

        return cov_config_upd

    def predict(
        self,
        params: linearmodels.typing.data.ArrayLike,
        *,
        exog: PanelDataLike | None = None,
        data: PanelDataLike | None = None,
        eval_env: int = 1,
        context: Mapping[str, Any] | None = None,
    ) -> DataFrame:
        """
        Predict values for additional data.

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1).
        exog : PanelData, DataFrame, Series, ndarray or DataArray, optional
            Exogenous regressors (nobs by nvar). The containers are those of
            ``exog`` in the model. A list or a tuple raises a ``TypeError`` that
            names ``exog``.
        data : DataFrame, optional
            Values to use when making predictions from a model constructed
            from a formula.
        eval_env : int, optional
            Depth of the stack to use when evaluating formulas. The default is 1.
            It is not used if `context` is provided.
        context : Mapping of str to object, optional
            The variables that the formula can use, by name. If not provided, it
            is captured at the depth `eval_env`.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters.

        Raises
        ------
        ValueError
            If `data` is provided and the model was not created using a formula,
            if both `exog` and `data` are provided, if neither is provided, or
            if `exog` does not have as many columns as there are parameters.
        TypeError
            If `exog` is a container that is not supported, e.g., a list or a tuple.

        Notes
        -----
        .. versionchanged:: 7.1

           Calling without `exog` and `data` raises a ``ValueError``. It raised
           an ``AssertionError``.

        If `data` is not None, then `exog` must be None.
        Predictions from models constructed using formulas can
        be computed using either `exog`, which will treat these are
        arrays of values corresponding to the formula-processed data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if data is not None and self.formula is None:
            raise ValueError(
                "Unable to use data when the model was not created using a formula."
            )
        if data is not None and exog is not None:
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog or data, but not both."
            )
        if data is None and exog is None:
            raise ValueError("One of exog or data must be provided.")
        if exog is not None:
            exog = _panel_data(exog, "exog", "x").dataframe
        else:
            assert self._formula is not None
            assert data is not None
            if context is None:
                context = capture_context(eval_env)
            parser = PanelFormulaParser(self._formula, data, context=context)
            exog = parser.exog
        x = exog.values
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        if x.shape[1] != params.shape[0]:
            raise ValueError(
                EXOG_PREDICT_MSG.format(
                    x_shape=x.shape[1], params_shape=params.shape[0]
                )
            )

        pred = DataFrame(x @ params, index=exog.index, columns=["predictions"])

        return pred


class PooledOLS(_PanelModelBase):
    r"""
    Pooled coefficient estimator for panel data.

    Parameters
    ----------
    dependent : PanelData, DataFrame, Series, ndarray or DataArray
        Dependent (left-hand-side) variable (time by entity). A DataFrame or
        Series with a 2-level MultiIndex (entity, then time), a DataFrame with
        time as the index and entities as the columns, a 2-d (time by entity)
        NumPy array or DataArray, or a PanelData. A list, a tuple or any other
        container raises a ``TypeError`` that names ``dependent``, since the
        entities and time periods need labels or a layout.
    exog : PanelData, DataFrame, Series, ndarray or DataArray
        Exogenous or right-hand-side variables (variable by time by entity). The
        containers are those of ``dependent``, where a DataFrame has a column for
        each variable and an array or DataArray with several variables has 3
        dimensions (variable by time by entity). A list, a tuple or any other
        container raises a ``TypeError`` that names ``exog``.
    weights : array_like, optional
        Weights to use in estimation. Assumes the variance of the residual is
        proportional to the inverse of the weight, so that the residual times
        the square root of the weight is homoskedastic.
        Weights can be given for each observation, for each time period,
        for each entity or for each combination of time period and entity.
        The weights must be strictly positive and finite. They are rescaled to have a
        mean of 1. A NaN weight is missing, and its observation is dropped.
        Weights with labels must use the same labels, in the same order,
        as dependent or, if dependent has no labels, as exog. See
        :ref:`panel-input-alignment`.
        Dependent need not be balanced when the weights are given for each time
        period, entity or combination: the weights of the combinations that
        dependent does not have are not used.
        A 1-d array has a value for each row of dependent, in the order of the
        rows, if it has as many values as dependent has rows. It has a value for
        each time period if it has as many values as there are time periods, and
        a value for each entity if it has as many as there are entities. Weights
        with labels (a pandas object with an index that is not the default index)
        are read by their labels, so that a Series indexed by the time periods has
        a value for each time period, and one indexed by the entities has a value
        for each entity. Weights without labels (an array, a list, a tuple, or a
        Series or a DataFrame with one column that has the default index) are read
        by their number of values. If dependent is not balanced and has as many
        rows as time periods or entities, they have a value for each row. An
        ``AmbiguityError`` is raised if the numbers of time periods and entities
        are equal and the weights have that many values, unless they have no
        labels and a value for each row. Any other number of values raises a
        ``ValueError``.
    check_rank : bool, optional
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank. The default is True.

    Raises
    ------
    TypeError
        If `dependent` or `exog` is a container that is not supported, e.g., a list
        or a tuple, or if they have complex values or values that cannot be
        converted to float64.
    ValueError
        If `dependent`, `exog` or `weights` have different shapes or labels, if a
        combination of an entity and a time period appears more than once, if the
        weights are not strictly positive and finite or do not have a supported
        shape, or if `exog` does not have full column rank and `check_rank` is
        True.
    AmbiguityError
        If the numbers of time periods and entities are equal and 1-d weights have
        that many values, so that they could be a value for each time period or
        for each entity.

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}

    .. versionchanged:: 7.1

       ``weights`` for each time period, each entity or each time period and
       entity can be used if ``dependent`` is not balanced. Earlier versions
       raised "weights must have the same number of observations as dependent".

    .. versionchanged:: 7.1

       ``weights`` that are a 1-d array with a value for each row of ``dependent``
       can be used if ``dependent`` is not balanced. Earlier versions raised
       "Weights do not have a supported shape". An array with a value for each
       combination of a time period and an entity is not a value for each row if
       ``dependent`` is not balanced. It raised an error from pandas or NumPy, e.g.,
       "could not broadcast input array from shape (30,1) into shape (27,1)", and
       raises "Weights do not have a supported shape" now, which names the numbers
       of values that are accepted. If ``dependent`` is not balanced and has as many
       rows as time periods or entities, weights without labels (an array, or a
       Series with the default index) with that many values are a value for each
       row, and a Series indexed by the time periods or by the entities is a value
       for each time period or entity. Earlier versions raised "weights must have
       the same number of observations as dependent", or an ``AmbiguityError`` if
       the numbers of time periods and entities were equal. A DataFrame with one
       column and the default index is read as an array with a column is. It raised
       an error from pandas or NumPy if ``dependent`` was balanced, and "Weights do
       not have a supported shape" if it was not. A DataFrame with one column and
       an index that is not the default index is read as a Series with that index
       is, so that its labels decide. It raised "Weights do not have a supported
       shape".
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> None:
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> PooledOLS:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules.
        data : PanelData or DataFrame
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time. A
            list or a tuple raises a ``TypeError`` that names ``data``.
        weights : array_like, optional
            Weights to use in estimation. Assumes the variance of the residual is
            proportional to the inverse of the weight, so that the residual times
            the square root of the weight is homoskedastic.
            The weights must be strictly positive and finite. They are rescaled to have
            a mean of 1. A NaN weight is missing, and its observation is dropped.
        check_rank : bool, optional
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank. The default is True.

        Returns
        -------
        PooledOLS
            Model specified using the formula.

        Notes
        -----
        Unlike standard  formula syntax, it is necessary to explicitly include
        a constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import PooledOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = PooledOLS.from_formula("y ~ 1 + x1", panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data, context=capture_context(1))
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights, check_rank=check_rank)
        mod.formula = formula
        return mod

    def fit(
        self,
        *,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: (
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData
        ),
    ) -> PanelResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator. The default is "unadjusted". The names
            are not case sensitive. Supported covariance estimators are (see
            Notes):

            * "unadjusted", "conventional", "homoskedastic" - Homoskedastic
              estimator
            * "robust", "heteroskedastic" - Heteroskedastic estimator
            * "clustered" - Clustered estimator
            * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay estimator
            * "ac", "autocorrelated" - Autocorrelation robust estimator

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True. See Notes.
        **cov_config : bool, float, str, array_like, DataFrame or PanelData
            Additional covariance-specific options. The options that each
            covariance estimator supports are listed in Notes. A key that the
            estimator does not support raises a ``ValueError``.

        Returns
        -------
        PanelResults
            Estimation results.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` is not one of the supported names, or if it is
            "kernel" or "ac" and ``kernel`` is not one of the supported kernels.
            It is both a ``ValueError`` and a ``KeyError``.
        ValueError
            If a key of ``cov_config`` is not supported by the covariance
            estimator (this is checked before the model is estimated), if
            ``clusters`` does not have the shape of the data, if ``clusters`` has
            a missing value for an observation that the model uses, if
            ``bandwidth`` is not below the number of time periods, or if the model
            has no residual degrees of freedom.
        TypeError
            If the value of a key of ``cov_config`` does not have the type that the
            key needs, e.g., if ``kernel`` is not a str.

        Notes
        -----
        Five covariance estimators are supported:

        * "unadjusted", "conventional", "homoskedastic" - Assume the residuals are
          homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered" - One- or two-way clustering. The configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables. This can be
            a pandas object or PanelData with the index of ``dependent``, a 1-d
            array or list with a value for each row of ``dependent`` in the
            order of its rows, or an array or list with the layout of
            ``dependent`` (time by entity, or variable by time by entity).
            An array or list with the layout, or a DataFrame with the time
            periods as its index and the entities as its columns, has a value
            for each time period and entity. It is placed on the rows of
            ``dependent``, which need not be balanced: the values of the
            combinations that ``dependent`` does not have are not used. See
            :ref:`panel-input-alignment`.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables.
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters. The default is False.
          * ``cluster_time`` - Boolean flag indicating to use time clusters. The
            default is False.
          * ``group_debias`` - Boolean flag indicating whether to apply an
            adjustment for the number of clusters. It multiplies the
            covariance by g / (g - 1) * (n - 1) / n, where g is the number
            of distinct clusters and n is the number of observations. With
            two-way clustering, it is applied to each of the three one-way
            covariances that are combined. The default is False.

        * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay HAC estimator. The
          configuration options are:

          * ``kernel`` - One of the supported kernels: "bartlett", "newey-west",
            "parzen", "gallant", "qs", "quadratic-spectral" or "andrews". The names
            are not case sensitive. The default is "newey-west", which is
            Bartlett's kernel.
          * ``bandwidth`` - Bandwidth to use when computing the kernel. If
            not provided, a naive default is used. It must be below the number
            of time periods, and a ``ValueError`` is raised otherwise.

        * "ac", "autocorrelated" - Autocorrelation robust estimator, which allows
          for autocorrelation of the errors of an entity and not for correlation
          between entities. The configuration options are ``kernel`` and
          ``bandwidth``, as for "kernel". The lags of an entity are the positions of
          its observations, not calendar periods, and the bandwidth can be larger
          than the number of observations of an entity. See
          :class:`~linearmodels.panel.covariance.ACCovariance`.

        ``debiased=True``, which is the default, scales the covariance by
        n / (n - k), where k is the number of regressors, and uses the t and
        F distributions for inference.

        ``extra_df`` can be used with any of the covariance estimators. It is the
        number of degrees of freedom that the model uses beyond the regressors,
        which the covariance estimator always accounts for, whether or not
        ``debiased`` is True. The default is 0.

        ``debiased=True`` and ``group_debias=True`` give the clustered
        covariance of Stata's ``regress ..., vce(cluster ...)``.
        See :ref:`panel-small-sample`.

        .. versionchanged:: 7.1

           A ``kernel`` that is not one of the supported names raises an
           ``UnknownOptionError`` when the model is fit, for "kernel" and "ac".
           Earlier versions raised it when the covariance was first used, for
           example when ``std_errors`` of the results was read, so that ``fit``
           returned results that raised when the covariance was used.

        .. versionchanged:: 7.1

           ``clusters`` with the layout of ``dependent`` (time by entity, or
           variable by time by entity) can be used if ``dependent`` is not
           balanced. It raised ``IndexError: Boolean index has wrong length``.
           It is placed on the rows of ``dependent`` by time period and entity.
           Earlier versions paired it by position with the rows, which is right
           only if the rows are sorted by entity and then time: if they were sorted
           by time and then entity, an array was attached to the wrong
           observations without an error, and a pandas object with labels was
           rejected.

        .. versionchanged:: 7.1

           A ``ValueError`` is raised if ``clusters`` has a missing value for an
           observation that the model uses, as in the IV and system models. Earlier
           versions made the observations with a missing value a cluster of their
           own, so that the standard errors changed without an error. The clusters
           of the observations that are dropped because a value of ``dependent``,
           ``exog`` or ``weights`` is missing are not used, and can be missing.

        Examples
        --------
        >>> from linearmodels import PooledOLS
        >>> from linearmodels.datasets import wage_panel
        >>> data = wage_panel.load().set_index(["nr", "year"])
        >>> y = data["lwage"]
        >>> x = data[["married", "union"]]
        >>> mod = PooledOLS(y, x)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)
        """
        cov_type = lower_option(cov_type)
        check_covariance_kwargs(self._cov_estimators, cov_type, cov_config)
        cov_config = self._setup_clusters(cov_config)
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = cast("linearmodels.typing.data.Float64Array", np.sqrt(w))
        wx = root_w * x
        wy = root_w * y

        params = _lstsq(wx, wy, rcond=None)[0]

        nobs = y.shape[0]
        df_model = x.shape[1]
        df_resid = nobs - df_model
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.pop("extra_df")
            assert isinstance(_extra_df, (str, int))
            extra_df = int(_extra_df)
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            wx,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )
        weps = wy - wx @ params
        index = self.dependent.index
        fitted = DataFrame(x @ params, index, ["fitted_values"])
        effects = DataFrame(
            np.full_like(np.asarray(fitted), np.nan), index, ["estimated_effects"]
        )
        eps = y - fitted.values
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        residual_ss = float(np.squeeze(weps.T @ weps))
        e = y
        if self._constant:
            e = e - (w * y).sum() / w.sum()

        total_ss = float(np.squeeze(w.T @ (e**2)))
        y_ss = float(np.squeeze(w.T @ (y**2)))
        r2 = r_squared(residual_ss, total_ss, y_ss, nobs)

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, weights=w
        )
        res.update(
            {
                "df_resid": df_resid,
                "df_model": df_model,
                "nobs": y.shape[0],
                "residual_ss": residual_ss,
                "total_ss": total_ss,
                "r2": r2,
                "wresids": weps,
                "resids": eps,
                "index": self.dependent.index,
                "fitted": fitted,
                "effects": effects,
                "idiosyncratic": idiosyncratic,
            }
        )

        return PanelResults(res)

    def predict(
        self,
        params: linearmodels.typing.data.ArrayLike,
        *,
        exog: PanelDataLike | None = None,
        data: PanelDataLike | None = None,
        eval_env: int = 1,
        context: Mapping[str, Any] | None = None,
    ) -> DataFrame:
        """
        Predict values for additional data.

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1).
        exog : PanelData, DataFrame, Series, ndarray or DataArray, optional
            Exogenous regressors (nobs by nvar). The containers are those of
            ``exog`` in the model. A list or a tuple raises a ``TypeError`` that
            names ``exog``.
        data : DataFrame, optional
            Values to use when making predictions from a model constructed
            from a formula.
        eval_env : int, optional
            Depth of the stack to use when evaluating formulas. The default is 1.
            It is not used if `context` is provided.
        context : Mapping of str to object, optional
            The variables that the formula can use, by name. If not provided, it
            is captured at the depth `eval_env`.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters.

        Raises
        ------
        ValueError
            If `data` is provided and the model was not created using a formula,
            if both `exog` and `data` are provided, if neither is provided, or
            if `exog` does not have as many columns as there are parameters.
        TypeError
            If `exog` is a container that is not supported, e.g., a list or a tuple.

        Notes
        -----
        .. versionchanged:: 7.1

           Calling without `exog` and `data` raises a ``ValueError``. It raised
           an ``AssertionError``.

        If `data` is not None, then `exog` must be None.
        Predictions from models constructed using formulas can
        be computed using either `exog`, which will treat these are
        arrays of values corresponding to the formula-processed data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if data is not None and self.formula is None:
            raise ValueError(
                "Unable to use data when the model was not created using a formula."
            )
        if data is not None and exog is not None:
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog or data, but not both."
            )
        if data is None and exog is None:
            raise ValueError("One of exog or data must be provided.")
        if exog is not None:
            exog = _panel_data(exog, "exog", "x").dataframe
        else:
            assert self._formula is not None
            assert data is not None
            if context is None:
                context = capture_context(eval_env)
            parser = PanelFormulaParser(self._formula, data, context=context)
            exog = parser.exog
        x = exog.values
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        if x.shape[1] != params.shape[0]:
            raise ValueError(
                EXOG_PREDICT_MSG.format(
                    x_shape=x.shape[1], params_shape=params.shape[0]
                )
            )
        pred = DataFrame(x @ params, index=exog.index, columns=["predictions"])

        return pred


class PanelOLS(_PanelModelBase):
    r"""
    One- and two-way fixed effects estimator for panel data.

    Parameters
    ----------
    dependent : PanelData, DataFrame, Series, ndarray or DataArray
        Dependent (left-hand-side) variable (time by entity). A DataFrame or
        Series with a 2-level MultiIndex (entity, then time), a DataFrame with
        time as the index and entities as the columns, a 2-d (time by entity)
        NumPy array or DataArray, or a PanelData. A list, a tuple or any other
        container raises a ``TypeError`` that names ``dependent``, since the
        entities and time periods need labels or a layout.
    exog : PanelData, DataFrame, Series, ndarray or DataArray
        Exogenous or right-hand-side variables (variable by time by entity). The
        containers are those of ``dependent``, where a DataFrame has a column for
        each variable and an array or DataArray with several variables has 3
        dimensions (variable by time by entity). A list, a tuple or any other
        container raises a ``TypeError`` that names ``exog``. It can have no
        variables, a DataFrame with the index of ``dependent`` and no columns or an
        array with 0 variables, if the model has effects. See the notes.
    weights : array_like, optional
        Weights to use in estimation. Assumes the variance of the residual is
        proportional to the inverse of the weight, so that the residual times
        the square root of the weight is homoskedastic.
        Weights can be given for each observation, for each time period,
        for each entity or for each combination of time period and entity.
        The weights must be strictly positive and finite. They are rescaled to have a
        mean of 1. A NaN weight is missing, and its observation is dropped.
        Weights with labels must use the same labels, in the same order,
        as dependent or, if dependent has no labels, as exog. See
        :ref:`panel-input-alignment`.
        Dependent need not be balanced when the weights are given for each time
        period, entity or combination: the weights of the combinations that
        dependent does not have are not used.
        A 1-d array has a value for each row of dependent, in the order of the
        rows, if it has as many values as dependent has rows. It has a value for
        each time period if it has as many values as there are time periods, and
        a value for each entity if it has as many as there are entities. Weights
        with labels (a pandas object with an index that is not the default index)
        are read by their labels, so that a Series indexed by the time periods has
        a value for each time period, and one indexed by the entities has a value
        for each entity. Weights without labels (an array, a list, a tuple, or a
        Series or a DataFrame with one column that has the default index) are read
        by their number of values. If dependent is not balanced and has as many
        rows as time periods or entities, they have a value for each row. An
        ``AmbiguityError`` is raised if the numbers of time periods and entities
        are equal and the weights have that many values, unless they have no
        labels and a value for each row. Any other number of values raises a
        ``ValueError``.
    entity_effects : bool, optional
        Flag whether to include entity (fixed) effects in the model. The default
        is False.
    time_effects : bool, optional
        Flag whether to include time effects in the model. The default is False.
    other_effects : array_like, optional
        Category codes to use for any effects that are not entity or time
        effects. Each variable is treated as an effect. An array or list with the
        layout of dependent (time by entity, or variable by time by entity), or a
        DataFrame with the time periods as its index and the entities as its
        columns, has a value for each time period and entity. It is placed on the
        rows of dependent, which need not be balanced: the values of the
        combinations that dependent does not have are not used (see
        :ref:`panel-input-alignment`). A 1-d array or list has
        a value for each row of dependent, in the order of the rows of dependent.
        An observation where any of the other effects is missing (NaN or None) is
        dropped, with the observations that have a missing value in dependent, exog
        or weights, and a ``MissingValueWarning`` is issued.
    singletons : bool, optional
        Flag indicating whether to keep singleton observations. If False, the
        observations that are alone in their group of an effect are dropped,
        repeatedly until there are none, and a ``SingletonWarning`` is issued.
        The weights are then scaled to have a mean of 1 in the observations that
        are retained. The default is True.

        .. versionchanged:: 7.1

           The weights were scaled before the singletons were dropped, so that
           their mean was not 1 in the retained observations and ``s2`` differed
           from that of the same model on data without the singletons.
    drop_absorbed : bool, optional
        Flag indicating whether to drop absorbed variables. If False, an
        ``AbsorbingEffectError`` is raised by ``fit`` for a model with an
        absorbed variable, and if True a variable is dropped with an
        ``AbsorbingEffectWarning``. The default is False.
    check_rank : bool, optional
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank. The default is True.

    Raises
    ------
    TypeError
        If `dependent` or `exog` is a container that is not supported, e.g., a list
        or a tuple, or if they have complex values or values that cannot be
        converted to float64.
    ValueError
        If `dependent`, `exog` or `weights` have different shapes or labels, if a
        combination of an entity and a time period appears more than once, if the
        weights are not strictly positive and finite or do not have a supported
        shape, or if `exog` does not have full column rank and `check_rank` is
        True. It is also raised if more than 2 effects are included, if
        two of the effects nest each other, if `other_effects` does not have as many
        entities and time periods as `dependent`, or if the model has no effects
        and `exog` has no columns.
    AmbiguityError
        If the numbers of time periods and entities are equal and 1-d weights have
        that many values, so that they could be a value for each time period or
        for each entity.

    Notes
    -----
    Many models can be estimated. The most common included entity effects and
    can be described

    .. math::

        y_{it} = \alpha_i + \beta^{\prime}x_{it} + \epsilon_{it}

    where :math:`\alpha_i` is included if ``entity_effects=True``.

    Time effect are also supported, which leads to a model of the form

    .. math::

        y_{it}= \gamma_t + \beta^{\prime}x_{it} + \epsilon_{it}

    where :math:`\gamma_i` is included if ``time_effects=True``.

    Both effects can be simultaneously used,

    .. math::

        y_{it}=\alpha_i + \gamma_t + \beta^{\prime}x_{it} + \epsilon_{it}

    Additionally , arbitrary effects can be specified using categorical variables.

    .. versionchanged:: 7.1

       ``weights`` for each time period, each entity or each time period and
       entity, and ``other_effects`` for each time period and entity, can be used
       if ``dependent`` is not balanced. Earlier versions raised "weights must
       have the same number of observations as dependent" for the weights, and an
       error for ``other_effects``. A value for each time period and entity is
       placed on the rows of ``dependent`` by time period and entity. Earlier
       versions paired ``other_effects`` by position with the rows, which is right
       only if the rows are sorted by entity and then time: if they were sorted by
       time and then entity, an array was attached to the wrong observations
       without an error, and a pandas object with labels was rejected.

    .. versionchanged:: 7.1

       ``weights`` that are a 1-d array with a value for each row of ``dependent``
       can be used if ``dependent`` is not balanced. Earlier versions raised
       "Weights do not have a supported shape". An array with a value for each
       combination of a time period and an entity is not a value for each row if
       ``dependent`` is not balanced. It raised an error from pandas or NumPy, e.g.,
       "could not broadcast input array from shape (30,1) into shape (27,1)", and
       raises "Weights do not have a supported shape" now, which names the numbers
       of values that are accepted. If ``dependent`` is not balanced and has as many
       rows as time periods or entities, weights without labels (an array, or a
       Series with the default index) with that many values are a value for each
       row, and a Series indexed by the time periods or by the entities is a value
       for each time period or entity. Earlier versions raised "weights must have
       the same number of observations as dependent", or an ``AmbiguityError`` if
       the numbers of time periods and entities were equal. A DataFrame with one
       column and the default index is read as an array with a column is. It raised
       an error from pandas or NumPy if ``dependent`` was balanced, and "Weights do
       not have a supported shape" if it was not. A DataFrame with one column and
       an index that is not the default index is read as a Series with that index
       is, so that its labels decide. It raised "Weights do not have a supported
       shape".

    If both ``entity_effect`` and ``time_effects`` are ``False``, and no other
    effects are included, the model reduces to :class:`PooledOLS`.

    Model supports at most 2 effects.  These can be entity-time, entity-other,
    time-other or 2 other.

    **Models without regressors**

    A model that only has effects can be estimated,

    .. math::

        y_{it} = \alpha_i + \gamma_t + \epsilon_{it}

    where either effect can be left out. The model is created with an ``exog`` that
    has no columns, for example ``pd.DataFrame(index=y.index)`` or an array with 0
    variables, or with a formula that has effects and no regressors, such as
    ``"y ~ EntityEffects"`` or ``"y ~ EntityEffects + TimeEffects"``. The formula
    gives the same results as ``exog`` without columns. The model has no constant, as
    a constant is a regressor (``"y ~ 1 + EntityEffects"`` has one). The other models,
    and ``PanelOLS`` without effects, raise a ``ValueError`` if ``exog`` has no
    columns. The results of a model without regressors are:

    * ``params``, ``std_errors``, ``tstats``, ``pvalues`` and ``cov`` are empty, and
      ``conf_int`` has no rows. The summary does not have a table of parameters, and
      ``wald_test`` raises a ``ValueError``.
    * ``estimated_effects`` are the means of the dependent variable of the entity
      (entity effects), of the time period (time effects), or the fitted values of
      the regression on the dummy variables of both effects. These are weighted means
      in a weighted model. There is no constant, so that the effects include the
      overall mean. ``fitted_values`` are 0, and ``resids`` and ``idiosyncratic`` are
      the dependent variable less its effects.
    * ``df_model`` is the number of effects, which is ``N`` for entity effects, ``T``
      for time effects and ``N + T - 1`` for both, and ``df_resid`` is ``nobs``
      less ``df_model``.
    * ``rsquared``, ``rsquared_within``, ``rsquared_between`` and ``rsquared_overall``
      are 0 when they are defined, since the effects are removed from the dependent
      variable and there is nothing else to explain it. The squared correlations
      (``corr_squared_within``, ``corr_squared_between`` and ``corr_squared_overall``)
      are 0 as well: the fitted values of the regressors are all 0, and these
      measures are set to 0 if the fitted values do not vary. ``rsquared_inclusive``
      is the R-squared of the effects, the share of the variation around the mean that
      they explain. An R-squared is not defined if the sum of squares that it divides
      is zero, for example ``rsquared_within`` if the dependent variable does not vary
      within entities. If the dependent variable is constant, there is no variation to
      explain and ``rsquared``, ``rsquared_within``, ``rsquared_inclusive`` and the
      squared correlations are NaN.
    * ``f_pooled`` is the test that the effects are zero. For entity effects it is
      the F-test of the one-way analysis of variance. ``f_statistic`` and
      ``f_statistic_robust``, which test the coefficients of the regressors, are
      ``InvalidTestStatistic`` since there are none.
    * ``loglik`` and ``s2`` are those of the regression on the dummy variables of the
      effects. ``variance_decomposition`` splits the second moment of the dependent
      variable, about 0 since there is no constant, into the variance of the
      residuals and the rest, which is attributed to the effects.

    .. versionchanged:: 7.1

       A model without regressors can be estimated. Earlier versions raised a
       ``ValueError`` from pandas, "No objects to concatenate", when ``exog`` had no
       columns or the formula had no regressors.

    A variable in ``exog`` is absorbed if the effects explain all of it. The
    algorithms that remove the effects are exact only up to rounding error, or up to
    a relative error of about 1e-8 if they are iterative, so a variable is treated
    as absorbed if less than 1e-5 of its norm is left after the effects are
    removed, whatever its units. It is treated in the same way if it is collinear
    with the variables before it once the effects are removed. An
    ``AbsorbingEffectError`` is raised for such a variable. If ``drop_absorbed`` is
    True, the variable is dropped and an ``AbsorbingEffectWarning`` is raised, and the
    constant is never dropped. The norm includes the mean of the variable, so a
    variable with a large mean and little other variation, less than 1e-5 of its norm,
    is also treated as absorbed. Subtract its mean to avoid this.

    .. versionchanged:: 7.1

       An observation with a missing value in ``other_effects`` is dropped with a
       ``MissingValueWarning``, as it is for a missing value in ``dependent``,
       ``exog`` or ``weights``. Earlier versions treated a missing value as a
       category of its own, and ``fit`` failed after it estimated the model with the
       error that the observations per group cannot be computed.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        entity_effects: bool = False,
        time_effects: bool = False,
        other_effects: PanelArrayLike | None = None,
        singletons: bool = True,
        drop_absorbed: bool = False,
        check_rank: bool = True,
    ) -> None:
        # The effects are needed while the data are validated, which drops the
        # observations that have a missing value in other_effects
        self._entity_effects = entity_effects
        self._time_effects = time_effects
        self._other_effects_input = other_effects
        self._other_effects_rows: PanelData | None = None
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)

        self._other_effect_cats: PanelData | None = None
        self._singletons = singletons
        self._other_effects = self._validate_effects()
        self._has_effect = entity_effects or time_effects or self.other_effects
        if self.exog.nvar == 0 and not self._has_effect:
            raise ValueError(
                "exog has no columns and the model has no effects, so there is "
                "nothing to estimate. A model without regressors needs entity, time "
                "or other effects."
            )
        self._drop_absorbed = drop_absorbed
        self._singleton_index = None
        self._drop_singletons()

    def _check_regressors(self) -> None:
        """Nothing is checked here: __init__ checks that a model has effects"""

    def _collect_effects(self) -> linearmodels.typing.NumericArray:
        if not self._has_effect:
            return np.empty((self.dependent.shape[0], 0))
        effects = []
        if self.entity_effects:
            effects.append(np.asarray(self.dependent.entity_ids).squeeze())
        if self.time_effects:
            effects.append(np.asarray(self.dependent.time_ids).squeeze())
        if self.other_effects:
            assert self._other_effect_cats is not None
            other = self._other_effect_cats.dataframe
            effects.extend([np.asarray(other[col]).squeeze() for col in other])
        return np.column_stack(effects)

    def _drop_singletons(self) -> None:
        if self._singletons or not self._has_effect:
            return
        effects = self._collect_effects()
        retain = in_2core_graph(effects)

        if np.all(retain):
            return

        nobs = retain.shape[0]
        ndropped = nobs - retain.sum()
        warnings.warn(
            f"{ndropped} singleton observations dropped",
            SingletonWarning,
            stacklevel=3,
        )
        drop = ~retain
        self._singleton_index = cast("linearmodels.typing.data.BoolArray", drop)
        self.dependent.drop(drop)
        self.exog.drop(drop)
        self.weights.drop(drop)
        # The weights have a mean of 1 in the rows that are retained
        w_df = self.weights.dataframe
        self.weights = PanelData(
            DataFrame(
                normalize_weights(np.asarray(w_df)),
                index=w_df.index,
                columns=w_df.columns,
            )
        )
        if self.other_effects:
            assert self._other_effect_cats is not None
            self._other_effect_cats.drop(drop)
        # Reverify exog matrix
        self._check_exog_rank()

    def __str__(self) -> str:
        out = super().__str__()
        additional = (
            "\nEntity Effects: {ee}, Time Effects: {te}, Num Other Effects: {oe}"
        )
        oe = 0
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.nvar
        additional = additional.format(
            ee=self.entity_effects, te=self.time_effects, oe=oe
        )
        out += additional
        return out

    def _missing_in_other_inputs(self) -> BoolArray | None:
        """
        Observations with a missing value in other_effects

        Returns
        -------
        ndarray or None
            A boolean array that is True for the observations of dependent where
            any of the other effects is missing. None if the model does not have
            other effects.
        """
        if self._other_effects_input is None:
            return None
        given = as_array_like(self._other_effects_input, "other_effects")
        effects = self._panel_rows(given, "other_effects", "OtherEffect")
        if effects.shape[1:] != self._original_shape[1:]:
            raise ValueError(
                "other_effects must have the same number of "
                "entities and time periods as dependent."
            )
        labelled = _is_labelled(self._other_effects_input)
        if _is_grid(given):
            # Placed on the rows of dependent by time period and entity
            effects = self._grid_to_rows(effects, labelled, "other_effects")
        else:
            self._check_labels("other_effects", effects.index, labelled)
        nrows = len(self._original_index)
        if effects.dataframe.shape[0] != nrows:
            raise ValueError(
                f"other_effects has {effects.dataframe.shape[0]} rows, but dependent "
                f"has {nrows}. It must have a value for each row of dependent."
            )
        self._other_effects_rows = effects
        return np.asarray(effects.isnull, dtype=bool)

    def _validate_effects(self) -> bool:
        """Check model effects"""
        effects = self._other_effects_rows
        if effects is None:
            return False
        # The rows of the input are not needed once the categories are made
        self._other_effects_rows = None

        num_effects = effects.nvar
        if num_effects + self.entity_effects + self.time_effects > 2:
            raise ValueError("At most two effects supported.")
        cats = {}
        effects_frame = effects.dataframe
        for col in effects_frame:
            cat = Categorical(effects_frame[col])
            # TODO: Bug in pandas-stube
            #  https://github.com/pandas-dev/pandas-stubs/issues/111
            cats[col] = cat.codes.astype(np.int64)
        cats_df = DataFrame(cats, index=effects_frame.index)
        cats_df = cats_df[effects_frame.columns]
        other_effects = PanelData(cats_df)
        other_effects.drop(~self.not_null)
        self._other_effect_cats = other_effects
        cats_array = other_effects.values2d
        nested = False
        nesting_effect = ""
        if cats_array.shape[1] == 2:
            nested = self._is_effect_nested(cats_array[:, [0]], cats_array[:, [1]])
            nested |= self._is_effect_nested(cats_array[:, [1]], cats_array[:, [0]])
            nesting_effect = "other effects"
        elif self.entity_effects:
            nested = self._is_effect_nested(
                cats_array[:, [0]], self.dependent.entity_ids
            )
            nested |= self._is_effect_nested(
                self.dependent.entity_ids, cats_array[:, [0]]
            )
            nesting_effect = "entity effects"
        elif self.time_effects:
            nested = self._is_effect_nested(cats_array[:, [0]], self.dependent.time_ids)
            nested |= self._is_effect_nested(
                self.dependent.time_ids, cats_array[:, [0]]
            )
            nesting_effect = "time effects"
        if nested:
            raise ValueError(
                "Included other effects nest or are nested "
                "by {effect}".format(effect=nesting_effect)
            )

        return True

    @property
    def entity_effects(self) -> bool:
        """Flag indicating whether entity effects are included."""
        return self._entity_effects

    @property
    def time_effects(self) -> bool:
        """Flag indicating whether time effects are included."""
        return self._time_effects

    @property
    def other_effects(self) -> bool:
        """Flag indicating whether other (generic) effects are included."""
        return self._other_effects

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        other_effects: PanelArrayLike | None = None,
        singletons: bool = True,
        drop_absorbed: bool = False,
        check_rank: bool = True,
    ) -> PanelOLS:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules with two special variable names, EntityEffects and
            TimeEffects which can be used to specify that the model should
            contain an entity effect or a time effect, respectively. The
            formula can have effects and no regressors, for example
            ``"y ~ EntityEffects"``, which is a model with only entity effects
            that gives the same results as ``exog`` without columns. See
            Examples.
        data : PanelData or DataFrame
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time. A
            list or a tuple raises a ``TypeError`` that names ``data``.
        weights : array_like, optional
            Weights to use in estimation. Assumes the variance of the residual is
            proportional to the inverse of the weight, so that the residual times
            the square root of the weight is homoskedastic.
            The weights must be strictly positive and finite. They are rescaled to have
            a mean of 1. A NaN weight is missing, and its observation is dropped.
        other_effects : array_like, optional
            Category codes to use for any effects that are not entity or time
            effects. Each variable is treated as an effect. An observation where
            any of the other effects is missing is dropped, with a
            ``MissingValueWarning``.
        singletons : bool, optional
            Flag indicating whether to keep singleton observations. If False, the
            observations that are alone in their group of an effect are dropped,
            repeatedly until there are none, and a ``SingletonWarning`` is issued.
            The default is True.
        drop_absorbed : bool, optional
            Flag indicating whether to drop absorbed variables. If False, an
            ``AbsorbingEffectError`` is raised by ``fit`` for a model with an
            absorbed variable, and if True a variable is dropped with an
            ``AbsorbingEffectWarning``. The default is False.
        check_rank : bool, optional
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank. The default is True.

        Returns
        -------
        PanelOLS
            Model specified using the formula.

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = PanelOLS.from_formula("y ~ 1 + x1 + EntityEffects", panel_data.data)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)

        A model with effects and no regressors

        >>> mod = PanelOLS.from_formula("y ~ EntityEffects", panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data, context=capture_context(1))
        entity_effect = parser.entity_effect
        time_effect = parser.time_effect
        dependent, exog = parser.data
        mod = cls(
            dependent,
            exog,
            entity_effects=entity_effect,
            time_effects=time_effect,
            weights=weights,
            other_effects=other_effects,
            singletons=singletons,
            drop_absorbed=drop_absorbed,
            check_rank=check_rank,
        )
        mod.formula = formula
        return mod

    def _lsmr_path(
        self,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """Sparse implementation, works for all scenarios"""
        y = cast("linearmodels.typing.data.Float64Array", self.dependent.values2d)
        x = cast("linearmodels.typing.data.Float64Array", self.exog.values2d)
        w = cast("linearmodels.typing.data.Float64Array", self.weights.values2d)
        root_w = np.sqrt(w)
        wybar = root_w * (w.T @ y / w.sum())
        wy = root_w * y
        wx = root_w * x
        if not self._has_effect:
            y_effect, x_effect = np.zeros_like(wy), np.zeros_like(wx)
            return wy, wx, wybar, y_effect, x_effect

        wy_gm = wybar
        wx_gm = root_w * (w.T @ x / w.sum())
        root_w_sparse = csc_array(root_w)

        cats_l: list[
            linearmodels.typing.data.AnyIntArray | linearmodels.typing.data.Float64Array
        ] = []
        if self.entity_effects:
            cats_l.append(self.dependent.entity_ids)
        if self.time_effects:
            cats_l.append(self.dependent.time_ids)
        if self.other_effects:
            assert self._other_effect_cats is not None
            cats_l.append(self._other_effect_cats.values2d)
        cats = np.concatenate(cats_l, 1)

        wd, cond = dummy_matrix(cats, precondition=True)
        assert isinstance(wd, csc_array)
        if self._is_weighted:
            wd = wd.multiply(root_w_sparse)

        wx_mean_l = []
        for i in range(x.shape[1]):
            cond_mean = lsmr(wd, wx[:, i], atol=1e-8, btol=1e-8)[0]
            cond_mean /= cond
            wx_mean_l.append(cond_mean)
        wy_mean: linearmodels.typing.data.Float64Array | csc_array
        wy_mean = lsmr(wd, wy, atol=1e-8, btol=1e-8)[0]
        wy_mean /= cond
        wy_mean = wy_mean[:, None]

        wy_mean = csc_array(wy_mean)

        # Purge fitted, weighted values
        sp_cond = diags_array(cond, format="csc")
        if wx_mean_l:
            # There is nothing to purge if there are no regressors
            wx_mean = csc_array(np.column_stack(wx_mean_l))
            wx = wx - (wd @ sp_cond @ wx_mean).toarray()
        wy = wy - (wd @ sp_cond @ wy_mean).toarray()

        if self.has_constant:
            wy += wy_gm
            wx += wx_gm
        else:
            wybar = np.array(0.0)

        y_effects = y - wy / root_w
        x_effects = x - wx / root_w

        return wy, wx, wybar, y_effects, x_effects

    def _slow_path(
        self,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """Frisch-Waugh-Lovell implementation, works for all scenarios"""
        w = cast("linearmodels.typing.data.Float64Array", self.weights.values2d)
        root_w = np.sqrt(w)

        y = root_w * cast(
            "linearmodels.typing.data.Float64Array", self.dependent.values2d
        )
        x = root_w * cast("linearmodels.typing.data.Float64Array", self.exog.values2d)
        if not self._has_effect:
            ybar = root_w @ _lstsq(root_w, y, rcond=None)[0]
            y_effect, x_effect = np.zeros_like(y), np.zeros_like(x)
            return y, x, ybar, y_effect, x_effect

        drop_first = self._constant
        d_l = []
        if self.entity_effects:
            d_l.append(self.dependent.dummies("entity", drop_first=drop_first).values)
            drop_first = True
        if self.time_effects:
            d_l.append(self.dependent.dummies("time", drop_first=drop_first).values)
            drop_first = True
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                dummies = get_dummies(oe[c], drop_first=drop_first).astype(np.float64)
                d_l.append(dummies.values)
                drop_first = True

        d = np.column_stack(d_l)
        wd = root_w * d
        if self.has_constant:
            wd -= root_w * (w.T @ d / w.sum())
            z = np.ones_like(root_w)
            d -= z * (z.T @ d / z.sum())

        x_mean = _lstsq(wd, x, rcond=None)[0]
        y_mean = _lstsq(wd, y, rcond=None)[0]

        # Save fitted unweighted effects to use in eps calculation
        x_effects = d @ x_mean
        y_effects = d @ y_mean

        # Purge fitted, weighted values
        x = x - wd @ x_mean
        y = y - wd @ y_mean

        ybar = root_w @ _lstsq(root_w, y, rcond=None)[0]
        return y, x, ybar, y_effects, x_effects

    def _choose_twoway_algo(self) -> bool:
        if not (self.entity_effects and self.time_effects):
            return False
        nentity, nobs = self.dependent.nentity, self.dependent.nobs
        nreg = min(nentity, nobs)
        if nreg < self.exog.shape[1]:
            return False
        # MiB
        reg_size = 8 * nentity * nobs * nreg // 2**20
        low_memory = reg_size > 2**10
        if low_memory:

            warnings.warn(
                "Using low-memory algorithm to estimate two-way model. Explicitly set "
                "low_memory=True to silence this message.  Set low_memory=False to use "
                "the standard algorithm that creates dummy variables for the smaller "
                "of the number of entities or number of time periods.",
                MemoryWarning,
                stacklevel=3,
            )
        return low_memory

    def _fast_path(self, low_memory: bool) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """Dummy-variable free estimation without weights"""
        _y = self.dependent.values2d
        _x = self.exog.values2d
        ybar = np.asarray(_y.mean(0))

        if not self._has_effect:
            return _y, _x, ybar

        y_gm = ybar
        x_gm = _x.mean(0)

        y = self.dependent
        x = self.exog

        if self.other_effects:
            assert self._other_effect_cats is not None
            groups = self._other_effect_cats
            if self.entity_effects or self.time_effects:
                groups = groups.copy()
                if self.entity_effects:
                    effect = self.dependent.entity_ids
                else:
                    effect = self.dependent.time_ids
                col = ensure_unique_column("additional.effect", groups.dataframe)
                groups.dataframe[col] = effect
            y = cast("PanelData", y.general_demean(groups))
            x = cast("PanelData", x.general_demean(groups))
        elif self.entity_effects and self.time_effects:
            y = cast("PanelData", y.demean("both", low_memory=low_memory))
            x = cast("PanelData", x.demean("both", low_memory=low_memory))
        elif self.entity_effects:
            y = cast("PanelData", y.demean("entity"))
            x = cast("PanelData", x.demean("entity"))
        else:  # self.time_effects
            y = cast("PanelData", y.demean("time"))
            x = cast("PanelData", x.demean("time"))

        y_arr = y.values2d
        x_arr = x.values2d

        if self.has_constant:
            y_arr = y_arr + y_gm
            x_arr = x_arr + x_gm
        else:
            ybar = np.asarray(0.0)

        return y_arr, x_arr, ybar

    def _weighted_fast_path(self, low_memory: bool) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """Dummy-variable free estimation with weights"""
        y_arr = self.dependent.values2d
        x_arr = self.exog.values2d
        w = self.weights.values2d
        root_w = cast("linearmodels.typing.data.Float64Array", np.sqrt(w))
        wybar = root_w * (w.T @ y_arr / w.sum())

        if not self._has_effect:
            wy_arr = root_w * self.dependent.values2d
            wx_arr = root_w * self.exog.values2d
            y_effect, x_effect = np.zeros_like(wy_arr), np.zeros_like(wx_arr)
            return wy_arr, wx_arr, wybar, y_effect, x_effect

        wy_gm = wybar
        wx_gm = root_w * (w.T @ x_arr / w.sum())

        y = self.dependent
        x = self.exog

        if self.other_effects:
            assert self._other_effect_cats is not None
            groups = self._other_effect_cats
            if self.entity_effects or self.time_effects:
                groups = groups.copy()
                if self.entity_effects:
                    effect = self.dependent.entity_ids
                else:
                    effect = self.dependent.time_ids
                col = ensure_unique_column("additional.effect", groups.dataframe)
                groups.dataframe[col] = effect
            wy = y.general_demean(groups, weights=self.weights)
            wx = x.general_demean(groups, weights=self.weights)
        elif self.entity_effects and self.time_effects:
            wy = cast(
                "PanelData",
                y.demean("both", weights=self.weights, low_memory=low_memory),
            )
            wx = cast(
                "PanelData",
                x.demean("both", weights=self.weights, low_memory=low_memory),
            )
        elif self.entity_effects:
            wy = cast("PanelData", y.demean("entity", weights=self.weights))
            wx = cast("PanelData", x.demean("entity", weights=self.weights))
        else:  # self.time_effects
            wy = cast("PanelData", y.demean("time", weights=self.weights))
            wx = cast("PanelData", x.demean("time", weights=self.weights))

        wy_arr = wy.values2d
        wx_arr = wx.values2d

        if self.has_constant:
            wy_arr += wy_gm
            wx_arr += wx_gm
        else:
            wybar = np.array(0.0)

        wy_effects = y.values2d - wy_arr / root_w
        wx_effects = x.values2d - wx_arr / root_w

        return wy_arr, wx_arr, wybar, wy_effects, wx_effects

    def _info(self) -> tuple[Series, Series, DataFrame | None]:
        """Information about model effects and panel structure"""

        entity_info, time_info, other_info = super()._info()

        if self.other_effects:
            other_info_values: list[Series] = []
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                name = "Observations per group (" + str(c) + ")"
                other_info_values.append(
                    panel_structure_stats(oe[c].values.astype(np.int32), name)
                )
            other_info = DataFrame(other_info_values)

        return entity_info, time_info, other_info

    @staticmethod
    def _is_effect_nested(
        effects: linearmodels.typing.NumericArray,
        clusters: linearmodels.typing.NumericArray,
    ) -> bool:
        """
        Determine whether the effects are nested by the covariance clusters

        An effect is nested if each of its groups is in one group of every one of
        the cluster variables, so that the result does not depend on their order.
        """
        for e in effects.T:
            _e = (e - e.min()).astype(np.int64)
            e_count = len(np.unique(_e))
            for c in clusters.T:
                _c = (c - c.min()).astype(np.int64)
                cmax = _c.max()
                ec = _e * (cmax + 1) + _c
                if len(np.unique(ec)) != e_count:
                    return False
        return True

    def _determine_df_adjustment(
        self,
        cov_type: str,
        **cov_config: (
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData
        ),
    ) -> bool:
        if cov_type != "clustered" or not self._has_effect:
            return True
        num_effects = self.entity_effects + self.time_effects
        if self.other_effects:
            assert self._other_effect_cats is not None
            num_effects += self._other_effect_cats.shape[1]

        clusters = cov_config.get("clusters", None)
        if clusters is None:  # No clusters
            return True

        effects = self._collect_effects()
        if num_effects == 1:
            return not self._is_effect_nested(
                effects, cast("linearmodels.typing.data.IntArray", clusters)
            )
        return True  # Default case for 2-way -- not completely clear

    def fit(
        self,
        *,
        use_lsdv: bool = False,
        use_lsmr: bool = False,
        low_memory: bool | None = None,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        auto_df: bool = True,
        count_effects: bool = True,
        **cov_config: (
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData
        ),
    ) -> PanelEffectsResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        use_lsdv : bool, optional
            Flag indicating to use the Least Squares Dummy Variable estimator
            to eliminate effects. The default is False, which uses only means
            and does not require constructing dummy variables for each effect.
        use_lsmr : bool, optional
            Flag indicating to use LSDV with the Sparse Equations and Least
            Squares estimator to eliminate the fixed effects. It takes
            precedence over ``use_lsdv``. The default is False.
        low_memory : bool, optional
            Flag indicating whether to use a low-memory algorithm when a model
            contains two-way fixed effects. If not provided, the choice is taken
            automatically, and the low memory algorithm is used if the
            required dummy variable array is both larger than the array of
            regressors in the model and requires more than 1 GiB.
        cov_type : str, optional
            Name of covariance estimator. The default is "unadjusted". The names
            are not case sensitive. Supported covariance estimators are (see
            Notes):

            * "unadjusted", "conventional", "homoskedastic" - Homoskedastic
              estimator
            * "robust", "heteroskedastic" - Heteroskedastic estimator
            * "clustered" - Clustered estimator
            * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay estimator
            * "ac", "autocorrelated" - Autocorrelation robust estimator

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True. The
            estimated effects are counted in the degrees of freedom whether
            or not ``debiased`` is True. See Notes.
        auto_df : bool, optional
            Flag indicating that the treatment of estimated effects in degree
            of freedom adjustment is automatically handled. This is useful
            since clustered standard errors that are clustered using the same
            variable as an effect do not require degree of freedom correction
            while other estimators such as the unadjusted covariance do. With
            two cluster variables, the effect has to be nested in both of them,
            which means that each of its groups is in one group of each variable.
            The default is True.
        count_effects : bool, optional
            Flag indicating that the covariance estimator should be adjusted
            to account for the estimation of effects in the model. Only used
            if ``auto_df=False``. The default is True.
        **cov_config : bool, float, str, array_like, DataFrame or PanelData
            Additional covariance-specific options. The options that each
            covariance estimator supports are listed in Notes. A key that the
            estimator does not support raises a ``ValueError``.

        Returns
        -------
        PanelEffectsResults
            Estimation results.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` is not one of the supported names, or if it is
            "kernel" or "ac" and ``kernel`` is not one of the supported kernels.
            It is both a ``ValueError`` and a ``KeyError``.
        AbsorbingEffectError
            If the effects absorb one or more variables of ``exog``, or make them
            collinear, and ``drop_absorbed`` is False in the model.
        ValueError
            If a key of ``cov_config`` is not supported by the covariance
            estimator (this is checked before the model is estimated), if
            ``clusters`` does not have the shape of the data, if ``clusters`` has
            a missing value for an observation that the model uses, if
            ``bandwidth`` is not below the number of time periods, if the model
            has no residual degrees of freedom, or if all of the variables of
            ``exog`` are absorbed.
        TypeError
            If the value of a key of ``cov_config`` does not have the type that the
            key needs, e.g., if ``kernel`` is not a str.

        Warns
        -----
        AbsorbingEffectWarning
            If variables are dropped since they are absorbed by the effects, which
            happens if ``drop_absorbed`` is True in the model.
        MemoryWarning
            If ``low_memory`` is not provided and the low-memory algorithm is used
            for a model with entity and time effects.

        Notes
        -----
        Five covariance estimators are supported:

        * "unadjusted", "conventional", "homoskedastic" - Assume the residuals are
          homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered" - One- or two-way clustering. The configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables. This can be
            a pandas object or PanelData with the index of ``dependent``, a 1-d
            array or list with a value for each row of ``dependent`` in the
            order of its rows, or an array or list with the layout of
            ``dependent`` (time by entity, or variable by time by entity).
            An array or list with the layout, or a DataFrame with the time
            periods as its index and the entities as its columns, has a value
            for each time period and entity. It is placed on the rows of
            ``dependent``, which need not be balanced: the values of the
            combinations that ``dependent`` does not have are not used. See
            :ref:`panel-input-alignment`.
            Clusters should be integer valued, although other types will
            be coerced to integer values by treating as categorical variables.
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters. The default is False.
          * ``cluster_time`` - Boolean flag indicating to use time clusters. The
            default is False.
          * ``group_debias`` - Boolean flag indicating whether to apply an
            adjustment for the number of clusters. It multiplies the
            covariance by g / (g - 1) * (n - 1) / n, where g is the number
            of distinct clusters and n is the number of observations. With
            two-way clustering, it is applied to each of the three one-way
            covariances that are combined. The default is False.

        * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay HAC estimator. The
          configuration options are:

          * ``kernel`` - One of the supported kernels: "bartlett", "newey-west",
            "parzen", "gallant", "qs", "quadratic-spectral" or "andrews". The names
            are not case sensitive. The default is "newey-west", which is
            Bartlett's kernel.
          * ``bandwidth`` - Bandwidth to use when computing the kernel. If
            not provided, a naive default is used. It must be below the number
            of time periods, and a ``ValueError`` is raised otherwise.

        * "ac", "autocorrelated" - Autocorrelation robust estimator, which allows
          for autocorrelation of the errors of an entity and not for correlation
          between entities. The configuration options are ``kernel`` and
          ``bandwidth``, as for "kernel". The lags of an entity are the positions of
          its observations, not calendar periods, and the bandwidth can be larger
          than the number of observations of an entity. See
          :class:`~linearmodels.panel.covariance.ACCovariance`.

        ``debiased=True``, which is the default, scales the covariance by
        n / (n - k), where k is the number of regressors, and uses the t and
        F distributions for inference.

        ``debiased=True`` and ``group_debias=True`` give the clustered
        covariance of Stata's ``xtreg, fe vce(cluster ...)``.
        See :ref:`panel-small-sample`.

        PanelOLS counts the effects that it estimates in the degrees of
        freedom, which is ``extra_df`` of the covariance estimator. If d is
        the number of effects, the covariance is scaled by n / (n - d - k)
        if ``debiased`` is True and by n / (n - d) if it is False. If
        ``auto_df`` is True, the effects are not counted when the model has
        a single effect that is nested in the clusters. Use ``auto_df=False``
        and ``count_effects`` to choose. Since ``extra_df`` is set from the
        effects, it is not a keyword argument of ``fit``.

        .. versionchanged:: 7.1

           ``extra_df`` raises a ``ValueError`` at the start of ``fit``. It
           raised a ``TypeError`` after the parameters were estimated.

        An effect is nested in a cluster variable if each of its groups is in one
        group of the variable, for example entity effects with
        ``cluster_entity=True``, or entity effects with clusters that are the
        industries of the entities. It is nested in the clusters if it is nested
        in every one of the cluster variables, so that the covariance does not
        depend on the order of the variables. Entity effects are nested in the
        two-way clusters (entity, industry) if every entity is in one industry,
        and they are not nested in the clusters (entity, time), as an entity has
        observations in several periods. They are counted in the degrees of
        freedom then. The effects are always counted if the model has more than
        one effect or the covariance is not clustered. See
        :ref:`panel-small-sample`.

        .. versionchanged:: 7.1

           With two cluster variables the effect has to be nested in every one
           of them. Earlier versions kept the result for the last variable only, so
           that the standard errors depended on the order of the variables, and the
           effects were not counted if they were nested in the last variable and
           not in the first.

        .. versionchanged:: 7.1

           A ``kernel`` that is not one of the supported names raises an
           ``UnknownOptionError`` when the model is fit, for "kernel" and "ac".
           Earlier versions raised it when the covariance was first used, for
           example when ``std_errors`` of the results was read, so that ``fit``
           returned results that raised when the covariance was used.

        .. versionchanged:: 7.1

           ``clusters`` with the layout of ``dependent`` (time by entity, or
           variable by time by entity) can be used if ``dependent`` is not
           balanced. It raised ``IndexError: Boolean index has wrong length``.
           It is placed on the rows of ``dependent`` by time period and entity.
           Earlier versions paired it by position with the rows, which is right
           only if the rows are sorted by entity and then time: if they were sorted
           by time and then entity, an array was attached to the wrong
           observations without an error, and a pandas object with labels was
           rejected.

        .. versionchanged:: 7.1

           A ``ValueError`` is raised if ``clusters`` has a missing value for an
           observation that the model uses, as in the IV and system models. Earlier
           versions made the observations with a missing value a cluster of their
           own, so that the standard errors changed without an error. The clusters
           of the observations that are dropped because a value of ``dependent``,
           ``exog``, ``weights`` or ``other_effects`` is missing are not used, and
           can be missing.

        Examples
        --------
        >>> from linearmodels import PanelOLS
        >>> from linearmodels.datasets import wage_panel
        >>> data = wage_panel.load().set_index(["nr", "year"])
        >>> y = data["lwage"]
        >>> x = data[["married", "union"]]
        >>> mod = PanelOLS(y, x, entity_effects=True)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)
        """
        cov_type = lower_option(cov_type)
        # extra_df is set from the effects that are estimated
        check_covariance_kwargs(
            self._cov_estimators, cov_type, cov_config, fit_options=()
        )
        # Check clusters if singletons were removed
        cov_config = self._setup_clusters(cov_config)

        weighted = np.any(self.weights.values2d != 1.0)

        if use_lsmr:
            y, x, ybar, y_effects, x_effects = self._lsmr_path()
        elif use_lsdv:
            y, x, ybar, y_effects, x_effects = self._slow_path()
        else:
            low_memory = (
                self._choose_twoway_algo() if low_memory is None else low_memory
            )
            if not weighted:
                y, x, ybar = self._fast_path(low_memory=low_memory)
                y_effects = np.array([0.0])
                x_effects = np.zeros(x.shape)
            else:
                y, x, ybar, y_effects, x_effects = self._weighted_fast_path(
                    low_memory=low_memory
                )

        neffects = 0
        drop_first = self.has_constant
        if self.entity_effects:
            neffects += self.dependent.nentity - drop_first
            drop_first = True
        if self.time_effects:
            neffects += self.dependent.nobs - drop_first
            drop_first = True
        if self.other_effects:
            assert self._other_effect_cats is not None
            oe = self._other_effect_cats.dataframe
            for c in oe:
                neffects += oe[c].nunique() - drop_first
                drop_first = True

        has_effects = self.entity_effects or self.time_effects or self.other_effects
        if has_effects and x.shape[1] > 0:
            # The regressors before the effects were removed set the scale of each
            # column, so the check does not depend on the units of the variables
            x_orig = np.sqrt(self.weights.values2d) * self.exog.values2d
            if not self._drop_absorbed:
                check_absorbed(x, [str(var) for var in self.exog.vars], x_orig)
            else:
                # TODO: Need to special case the constant here when determining which
                #  to retain since we always want to retain the constant if present
                retain = not_absorbed(x, self._constant, self._constant_index, x_orig)
                if not retain:
                    raise ValueError(
                        "All columns in exog have been fully absorbed by the included"
                        " effects. This model cannot be estimated."
                    )
                if len(retain) != x.shape[1]:
                    drop = set(range(x.shape[1])).difference(retain)
                    dropped = ", ".join([str(self.exog.vars[i]) for i in drop])

                    warnings.warn(
                        absorbing_warn_msg.format(absorbed_variables=dropped),
                        AbsorbingEffectWarning,
                        stacklevel=2,
                    )
                    x = x[:, retain]
                    # Update constant index loc
                    if self._constant:
                        assert isinstance(self._constant_index, int)
                        self._constant_index = int(
                            np.squeeze(
                                np.argwhere(np.array(retain) == self._constant_index)
                            )
                        )

                    # Adjust exog
                    self.exog = PanelData(self.exog.dataframe.iloc[:, retain])
                    x_effects = x_effects[:, retain]

        params = _lstsq(x, y, rcond=None)[0]
        nobs = self.dependent.dataframe.shape[0]
        df_model = x.shape[1] + neffects
        df_resid = nobs - df_model
        if auto_df:
            count_effects = self._determine_df_adjustment(cov_type, **cov_config)
        extra_df = neffects if count_effects else 0
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            y,
            x,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = y - x @ params
        eps = weps
        _y = self.dependent.values2d
        _x = self.exog.values2d
        if weighted:
            eps = (_y - y_effects) - (_x - x_effects) @ params
            if self.has_constant:
                # Correction since y_effects and x_effects @ params add mean
                w = self.weights.values2d
                eps -= (w * eps).sum() / w.sum()
        index = self.dependent.index
        fitted = DataFrame(_x @ params, index, ["fitted_values"])
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        eps_effects = _y - fitted.values

        sigma2_tot = float(np.squeeze(eps_effects.T @ eps_effects) / nobs)
        sigma2_eps = float(np.squeeze(eps.T @ eps) / nobs)
        sigma2_effects = sigma2_tot - sigma2_eps
        rho = sigma2_effects / sigma2_tot if sigma2_tot > 0.0 else 0.0

        resid_ss = float(np.squeeze(weps.T @ weps))
        if self.has_constant:
            mu = ybar
        else:
            mu = np.array([0.0])
        total_ss = float(np.squeeze((y - mu).T @ (y - mu)))

        root_w = cast(
            "linearmodels.typing.data.Float64Array", np.sqrt(self.weights.values2d)
        )
        y_ex = root_w * self.dependent.values2d
        # The size of the dependent variable, before the effects are removed
        y_ss = float(np.squeeze(y_ex.T @ y_ex))
        r2 = r_squared(resid_ss, total_ss, y_ss, nobs)
        mu_ex = np.array(0.0, dtype=float)
        if (
            self.has_constant
            or self.entity_effects
            or self.time_effects
            or self.other_effects
        ):
            mu_ex = root_w * ((root_w.T @ y_ex) / (root_w.T @ root_w))
        total_ss_ex_effect = float(np.squeeze((y_ex - mu_ex).T @ (y_ex - mu_ex)))
        r2_ex_effects = r_squared(resid_ss, total_ss_ex_effect, y_ss, nobs)

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, y, x, weights=self.weights.values2d
        )
        ######################################
        # Pooled f-stat
        ######################################
        if self.entity_effects or self.time_effects or self.other_effects:
            wy, wx = root_w * self.dependent.values2d, root_w * self.exog.values2d
            df_num, df_denom = (df_model - wx.shape[1]), df_resid
            if not self.has_constant:
                # Correction for when models does not have explicit constant
                wy -= root_w * _lstsq(root_w, wy, rcond=None)[0]
                wx -= root_w * _lstsq(root_w, wx, rcond=None)[0]
                df_num -= 1
            weps_pooled = wy - wx @ _lstsq(wx, wy, rcond=None)[0]
            resid_ss_pooled = float(np.squeeze(weps_pooled.T @ weps_pooled))
            name = "Pooled F-statistic"
            f_pooled: (
                WaldTestStatistic | InapplicableTestStatistic | InvalidTestStatistic
            )
            if df_num <= 0:
                # A single entity (or period) with a constant: the effect is the constant
                f_pooled = InapplicableTestStatistic(
                    reason="The effects do not use any degrees of freedom", name=name
                )
            elif df_denom <= 0:
                f_pooled = InvalidTestStatistic(
                    "The model has no residual degrees of freedom", name=name
                )
            elif resid_ss <= 0.0:
                f_pooled = InvalidTestStatistic(
                    "The residual sum of squares is 0", name=name
                )
            else:
                num = (resid_ss_pooled - resid_ss) / df_num
                denom = resid_ss / df_denom
                f_pooled = WaldTestStatistic(
                    num / denom,
                    "Effects are zero",
                    df_num,
                    df_denom=df_denom,
                    name=name,
                )
            res.update(f_pooled=f_pooled)
            effects = DataFrame(
                eps_effects - eps,
                columns=["estimated_effects"],
                index=self.dependent.index,
            )
        else:
            effects = DataFrame(
                np.zeros_like(eps),
                columns=["estimated_effects"],
                index=self.dependent.index,
            )

        res.update(
            {
                "df_resid": df_resid,
                "df_model": df_model,
                "nobs": y.shape[0],
                "residual_ss": resid_ss,
                "total_ss": total_ss,
                "wresids": weps,
                "resids": eps,
                "r2": r2,
                "entity_effects": self.entity_effects,
                "time_effects": self.time_effects,
                "other_effects": self.other_effects,
                "sigma2_eps": sigma2_eps,
                "sigma2_effects": sigma2_effects,
                "rho": rho,
                "r2_ex_effects": r2_ex_effects,
                "effects": effects,
                "fitted": fitted,
                "idiosyncratic": idiosyncratic,
            }
        )

        return PanelEffectsResults(res)


class BetweenOLS(_PanelModelBase):
    r"""
    Between estimator for panel data.

    Parameters
    ----------
    dependent : PanelData, DataFrame, Series, ndarray or DataArray
        Dependent (left-hand-side) variable (time by entity). A DataFrame or
        Series with a 2-level MultiIndex (entity, then time), a DataFrame with
        time as the index and entities as the columns, a 2-d (time by entity)
        NumPy array or DataArray, or a PanelData. A list, a tuple or any other
        container raises a ``TypeError`` that names ``dependent``, since the
        entities and time periods need labels or a layout.
    exog : PanelData, DataFrame, Series, ndarray or DataArray
        Exogenous or right-hand-side variables (variable by time by entity). The
        containers are those of ``dependent``, where a DataFrame has a column for
        each variable and an array or DataArray with several variables has 3
        dimensions (variable by time by entity). A list, a tuple or any other
        container raises a ``TypeError`` that names ``exog``.
    weights : array_like, optional
        Weights to use in estimation. Assumes the variance of the residual is
        proportional to the inverse of the weight, so that the residual times
        the square root of the weight is homoskedastic.
        Weights can be given for each observation, for each time period,
        for each entity or for each combination of time period and entity.
        The weights must be strictly positive and finite. They are rescaled to have a
        mean of 1. A NaN weight is missing, and its observation is dropped.
        Weights with labels must use the same labels, in the same order,
        as dependent or, if dependent has no labels, as exog. See
        :ref:`panel-input-alignment`.
        Dependent need not be balanced when the weights are given for each time
        period, entity or combination: the weights of the combinations that
        dependent does not have are not used.
        A 1-d array has a value for each row of dependent, in the order of the
        rows, if it has as many values as dependent has rows. It has a value for
        each time period if it has as many values as there are time periods, and
        a value for each entity if it has as many as there are entities. Weights
        with labels (a pandas object with an index that is not the default index)
        are read by their labels, so that a Series indexed by the time periods has
        a value for each time period, and one indexed by the entities has a value
        for each entity. Weights without labels (an array, a list, a tuple, or a
        Series or a DataFrame with one column that has the default index) are read
        by their number of values. If dependent is not balanced and has as many
        rows as time periods or entities, they have a value for each row. An
        ``AmbiguityError`` is raised if the numbers of time periods and entities
        are equal and the weights have that many values, unless they have no
        labels and a value for each row. Any other number of values raises a
        ``ValueError``.
    check_rank : bool, optional
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank. The default is True.

    Raises
    ------
    TypeError
        If `dependent` or `exog` is a container that is not supported, e.g., a list
        or a tuple, or if they have complex values or values that cannot be
        converted to float64.
    ValueError
        If `dependent`, `exog` or `weights` have different shapes or labels, if a
        combination of an entity and a time period appears more than once, if the
        weights are not strictly positive and finite or do not have a supported
        shape, or if `exog` does not have full column rank and `check_rank` is
        True.
    AmbiguityError
        If the numbers of time periods and entities are equal and 1-d weights have
        that many values, so that they could be a value for each time period or
        for each entity.

    Notes
    -----
    The model is given by

    .. math::

        \bar{y}_{i}=  \beta^{\prime}\bar{x}_{i}+\bar{\epsilon}_{i}

    where :math:`\bar{z}` is the time-average.

    .. versionchanged:: 7.1

       ``weights`` for each time period, each entity or each time period and
       entity can be used if ``dependent`` is not balanced. Earlier versions
       raised "weights must have the same number of observations as dependent".

    .. versionchanged:: 7.1

       ``weights`` that are a 1-d array with a value for each row of ``dependent``
       can be used if ``dependent`` is not balanced. Earlier versions raised
       "Weights do not have a supported shape". An array with a value for each
       combination of a time period and an entity is not a value for each row if
       ``dependent`` is not balanced. It raised an error from pandas or NumPy, e.g.,
       "could not broadcast input array from shape (30,1) into shape (27,1)", and
       raises "Weights do not have a supported shape" now, which names the numbers
       of values that are accepted. If ``dependent`` is not balanced and has as many
       rows as time periods or entities, weights without labels (an array, or a
       Series with the default index) with that many values are a value for each
       row, and a Series indexed by the time periods or by the entities is a value
       for each time period or entity. Earlier versions raised "weights must have
       the same number of observations as dependent", or an ``AmbiguityError`` if
       the numbers of time periods and entities were equal. A DataFrame with one
       column and the default index is read as an array with a column is. It raised
       an error from pandas or NumPy if ``dependent`` was balanced, and "Weights do
       not have a supported shape" if it was not. A DataFrame with one column and
       an index that is not the default index is read as a Series with that index
       is, so that its labels decide. It raised "Weights do not have a supported
       shape".
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> None:
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)
        self._cov_estimators = CovarianceManager(
            self.__class__.__name__,
            HomoskedasticCovariance,
            HeteroskedasticCovariance,
            ClusteredCovariance,
        )

    def _setup_clusters(
        self,
        cov_config: Mapping[
            str,
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData,
        ],
    ) -> dict[
        str,
        bool | float | str | linearmodels.typing.data.IntArray | DataFrame | PanelData,
    ]:
        """Return covariance estimator reformat clusters"""
        cov_config_upd = dict(cov_config)
        if "clusters" not in cov_config:
            return cov_config_upd

        clusters = cov_config.get("clusters", None)
        if clusters is not None:
            cluster_data = cast(
                "Union[linearmodels.typing.data.IntArray, DataFrame, PanelData]",
                clusters,
            )
            clusters_panel = self.reformat_clusters(cluster_data)
            cluster_max = np.nanmax(clusters_panel.values3d, axis=1)
            delta = cluster_max - np.nanmin(clusters_panel.values3d, axis=1)
            if np.any(delta != 0):
                raise ValueError("clusters must not vary within an entity")

            index = clusters_panel.panel.minor_axis
            reindex = clusters_panel.entities
            clusters_frame = DataFrame(
                cluster_max.T, index=index, columns=clusters_panel.vars
            )
            # TODO: Bug in pandas-stubs prevents using Hashable | None
            clusters_frame = clusters_frame.loc[reindex].astype(np.int64)
            cov_config_upd["clusters"] = clusters_frame

        return cov_config_upd

    def fit(
        self,
        *,
        reweight: bool = False,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: (
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData
        ),
    ) -> PanelResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        reweight : bool, optional
            Flag indicating to reweight observations if the input data is
            unbalanced using a WLS estimator. If weights are provided, these
            are accounted for when reweighting. Has no effect on balanced data.
            The default is False.
        cov_type : str, optional
            Name of covariance estimator. The default is "unadjusted". The names
            are not case sensitive. Supported covariance estimators are (see
            Notes):

            * "unadjusted", "conventional", "homoskedastic" - Homoskedastic
              estimator
            * "robust", "heteroskedastic" - Heteroskedastic estimator
            * "clustered" - Clustered estimator

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True. See Notes.
        **cov_config : bool, float, str, array_like, DataFrame or PanelData
            Additional covariance-specific options. The options that each
            covariance estimator supports are listed in Notes. A key that the
            estimator does not support raises a ``ValueError``.

        Returns
        -------
        PanelResults
            Estimation results.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` is not one of the supported names. It is both a
            ``ValueError`` and a ``KeyError``.
        ValueError
            If the regressors are collinear in the means of the entities, which are
            the observations of the regression, unless ``check_rank`` is False in
            the model. This happens if there are fewer entities than regressors, or
            if a regressor is a combination of the others only in the means, e.g.,
            the dummy variables of the periods and the constant in a balanced
            panel. It is also raised if a key of ``cov_config`` is not supported by
            the covariance estimator, if the clusters vary within an entity, if
            ``clusters`` has a missing value for an observation that the model
            uses, or if the model has no residual degrees of freedom.
        TypeError
            If the value of a key of ``cov_config`` does not have the type that the
            key needs.

        Notes
        -----
        Three covariance estimators are supported:

        * "unadjusted", "conventional", "homoskedastic" - Assume the residuals are
          homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered" - One- or two-way clustering. The configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables. This can be
            a pandas object or PanelData with the index of ``dependent``, a 1-d
            array or list with a value for each row of ``dependent`` in the
            order of its rows, or an array or list with the layout of
            ``dependent`` (time by entity, or variable by time by entity).
            An array or list with the layout, or a DataFrame with the time
            periods as its index and the entities as its columns, has a value
            for each time period and entity. It is placed on the rows of
            ``dependent``, which need not be balanced: the values of the
            combinations that ``dependent`` does not have are not used. See
            :ref:`panel-input-alignment`.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables.
          * ``group_debias`` - Boolean flag indicating whether to apply an
            adjustment for the number of clusters. It multiplies the
            covariance by g / (g - 1) * (n - 1) / n, where g is the number
            of distinct clusters and n is the number of observations. With
            two-way clustering, it is applied to each of the three one-way
            covariances that are combined. The default is False.

        When using a clustered covariance estimator, all cluster ids must be
        identical within an entity. The options ``cluster_entity`` and
        ``cluster_time`` are not supported, since the observations are the entities.

        ``debiased=True``, which is the default, scales the covariance by
        n / (n - k), where k is the number of regressors, and uses the t and
        F distributions for inference. See :ref:`panel-small-sample`.

        ``extra_df`` can be used with any of the covariance estimators. It is the
        number of degrees of freedom that the model uses beyond the regressors,
        which the covariance estimator always accounts for, whether or not
        ``debiased`` is True. The default is 0.

        .. versionchanged:: 7.1

           ``clusters`` with the layout of ``dependent`` (time by entity, or
           variable by time by entity) can be used if ``dependent`` is not
           balanced. It raised ``IndexError: Boolean index has wrong length``.
           It is placed on the rows of ``dependent`` by time period and entity.
           Earlier versions paired it by position with the rows, which is right
           only if the rows are sorted by entity and then time: if they were sorted
           by time and then entity, an array was attached to the wrong
           observations without an error, and a pandas object with labels was
           rejected.

        .. versionchanged:: 7.1

           A ``ValueError`` is raised if ``clusters`` has a missing value for an
           observation that the model uses, as in the IV and system models. Earlier
           versions ignored a missing number if another observation of the entity
           had a cluster, and reported the other cases as clusters that vary within
           an entity. The clusters of the observations that are dropped because a
           value of ``dependent``, ``exog`` or ``weights`` is missing are not used,
           and can be missing.

        Examples
        --------
        >>> from linearmodels import BetweenOLS
        >>> from linearmodels.datasets import wage_panel
        >>> data = wage_panel.load().set_index(["nr", "year"])
        >>> y = data["lwage"]
        >>> x = data[["married", "union"]]
        >>> mod = BetweenOLS(y, x)
        >>> res = mod.fit(cov_type='robust')
        """
        cov_type = lower_option(cov_type)
        check_covariance_kwargs(
            self._cov_estimators, cov_type, cov_config, cluster_options=()
        )
        cov_config = self._setup_clusters(cov_config)
        y, x, w = self._prepare_between()
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(y)
        else:
            root_w = cast("linearmodels.typing.data.Float64Array", np.sqrt(w))

        wx = root_w * x
        wy = root_w * y
        params, _, rank, _ = _lstsq(wx, wy, rcond=None)
        if self._check_rank and rank < x.shape[1]:
            raise ValueError(
                "The regressors are collinear in the means of the entities, which "
                f"are the {y.shape[0]} observations of the regression, so that the "
                "parameters are not identified. This happens if there are fewer "
                "entities than regressors, or if a regressor is a combination of "
                "the others only in the means, e.g., the dummy variables of the "
                "periods and the constant in a balanced panel. If you wish to "
                "proceed, you can set check_rank=False."
            )

        df_resid = y.shape[0] - x.shape[1]
        df_model = x.shape[1]
        nobs = y.shape[0]
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.pop("extra_df")
            assert isinstance(_extra_df, (str, int))
            extra_df = int(_extra_df)
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            wx,
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )
        weps = wy - wx @ params
        index = self.dependent.index
        fitted = DataFrame(self.exog.values2d @ params, index, ["fitted_values"])
        eps = y - x @ params
        effects = DataFrame(eps, self.dependent.entities, ["estimated_effects"])
        idx = cast("MultiIndex", fitted.index)
        entities = idx.levels[0][idx.codes[0]]
        effects = effects.loc[entities]
        effects.index = idx
        dep = self.dependent.dataframe
        fitted = fitted.reindex(dep.index)
        effects = effects.reindex(dep.index)
        idiosyncratic = DataFrame(
            np.asarray(dep) - np.asarray(fitted) - np.asarray(effects),
            dep.index,
            ["idiosyncratic"],
        )

        residual_ss = float(np.squeeze(weps.T @ weps))
        e = y
        if self._constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(np.squeeze(w.T @ (e**2)))
        y_ss = float(np.squeeze(w.T @ (y**2)))
        r2 = r_squared(residual_ss, total_ss, y_ss, nobs)

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, weights=w
        )
        res.update(
            {
                "df_resid": df_resid,
                "df_model": df_model,
                "nobs": nobs,
                "residual_ss": residual_ss,
                "total_ss": total_ss,
                "r2": r2,
                "wresids": weps,
                "resids": eps,
                "index": self.dependent.entities,
                "fitted": fitted,
                "effects": effects,
                "idiosyncratic": idiosyncratic,
            }
        )

        return PanelResults(res)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> BetweenOLS:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules.
        data : PanelData or DataFrame
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time. A
            list or a tuple raises a ``TypeError`` that names ``data``.
        weights : array_like, optional
            Weights to use in estimation. Assumes the variance of the residual is
            proportional to the inverse of the weight, so that the residual times
            the square root of the weight is homoskedastic.
            The weights must be strictly positive and finite. They are rescaled to have
            a mean of 1. A NaN weight is missing, and its observation is dropped.
        check_rank : bool, optional
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank. The default is True.

        Returns
        -------
        BetweenOLS
            Model specified using the formula.

        Notes
        -----
        Unlike standard  formula syntax, it is necessary to explicitly include
        a constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import BetweenOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = BetweenOLS.from_formula("y ~ 1 + x1", panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data, context=capture_context(1))
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights, check_rank=check_rank)
        mod.formula = formula
        return mod


class FirstDifferenceOLS(_PanelModelBase):
    r"""
    First difference model for panel data.

    Parameters
    ----------
    dependent : PanelData, DataFrame, Series, ndarray or DataArray
        Dependent (left-hand-side) variable (time by entity). A DataFrame or
        Series with a 2-level MultiIndex (entity, then time), a DataFrame with
        time as the index and entities as the columns, a 2-d (time by entity)
        NumPy array or DataArray, or a PanelData. A list, a tuple or any other
        container raises a ``TypeError`` that names ``dependent``, since the
        entities and time periods need labels or a layout.
    exog : PanelData, DataFrame, Series, ndarray or DataArray
        Exogenous or right-hand-side variables (variable by time by entity). The
        containers are those of ``dependent``, where a DataFrame has a column for
        each variable and an array or DataArray with several variables has 3
        dimensions (variable by time by entity). A list, a tuple or any other
        container raises a ``TypeError`` that names ``exog``.
    weights : array_like, optional
        Weights to use in estimation. Assumes the variance of the residual is
        proportional to the inverse of the weight, so that the residual times
        the square root of the weight is homoskedastic.
        Weights can be given for each observation, for each time period,
        for each entity or for each combination of time period and entity.
        The weights must be strictly positive and finite. They are rescaled to have a
        mean of 1. A NaN weight is missing, and its observation is dropped.
        Weights with labels must use the same labels, in the same order,
        as dependent or, if dependent has no labels, as exog. See
        :ref:`panel-input-alignment`.
        Dependent need not be balanced when the weights are given for each time
        period, entity or combination: the weights of the combinations that
        dependent does not have are not used.
        A 1-d array has a value for each row of dependent, in the order of the
        rows, if it has as many values as dependent has rows. It has a value for
        each time period if it has as many values as there are time periods, and
        a value for each entity if it has as many as there are entities. Weights
        with labels (a pandas object with an index that is not the default index)
        are read by their labels, so that a Series indexed by the time periods has
        a value for each time period, and one indexed by the entities has a value
        for each entity. Weights without labels (an array, a list, a tuple, or a
        Series or a DataFrame with one column that has the default index) are read
        by their number of values. If dependent is not balanced and has as many
        rows as time periods or entities, they have a value for each row. An
        ``AmbiguityError`` is raised if the numbers of time periods and entities
        are equal and the weights have that many values, unless they have no
        labels and a value for each row. Any other number of values raises a
        ``ValueError``.
    check_rank : bool, optional
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank. The default is True.

    Raises
    ------
    TypeError
        If `dependent` or `exog` is a container that is not supported, e.g., a list
        or a tuple, or if they have complex values or values that cannot be
        converted to float64.
    ValueError
        If `dependent`, `exog` or `weights` have different shapes or labels, if a
        combination of an entity and a time period appears more than once, if the
        weights are not strictly positive and finite or do not have a supported
        shape, or if `exog` does not have full column rank and `check_rank` is
        True. It is also raised if `exog` has a constant, since it is differenced
        away, or if the panel has fewer than 2 time periods.
    AmbiguityError
        If the numbers of time periods and entities are equal and 1-d weights have
        that many values, so that they could be a value for each time period or
        for each entity.

    Notes
    -----
    The model is given by

    .. math::

        \Delta y_{it}=\beta^{\prime}\Delta x_{it}+\Delta\epsilon_{it}

    The differences are between adjacent periods. An observation is differenced with
    the observation of the same entity in the preceding time label of the panel. The
    labels of the panel are the distinct time labels that occur for any entity, in
    order. If the entity has no observation in the preceding label, the observation
    has no difference and is not used. This is the case for the first observation of
    an entity and for an observation that follows a gap, one or more periods in which
    the entity is not observed although another entity is. The model is fit to the
    differences that remain, and ``nobs`` is their number.

    Other packages can difference differently. plm differences consecutive
    observations of an entity, also across a gap, so that a difference can span
    several periods and the number of differences and the estimates are different if
    there are gaps. In a panel where about a quarter of the periods are missing, plm
    had 210 differences where this model has 167 and a coefficient of 0.474 where this
    model has 0.562. The estimates are the same if no entity has a gap. See
    :ref:`panel-ac-lags`.

    .. versionchanged:: 7.1

       The observations are differenced in the order of time. Earlier versions
       differenced against non-adjacent periods if the first entity enters late or
       has a gap.

    .. versionchanged:: 7.1

       ``weights`` for each time period, each entity or each time period and
       entity can be used if ``dependent`` is not balanced. Earlier versions
       raised "weights must have the same number of observations as dependent".

    .. versionchanged:: 7.1

       ``weights`` that are a 1-d array with a value for each row of ``dependent``
       can be used if ``dependent`` is not balanced. Earlier versions raised
       "Weights do not have a supported shape". An array with a value for each
       combination of a time period and an entity is not a value for each row if
       ``dependent`` is not balanced. It raised an error from pandas or NumPy, e.g.,
       "could not broadcast input array from shape (30,1) into shape (27,1)", and
       raises "Weights do not have a supported shape" now, which names the numbers
       of values that are accepted. If ``dependent`` is not balanced and has as many
       rows as time periods or entities, weights without labels (an array, or a
       Series with the default index) with that many values are a value for each
       row, and a Series indexed by the time periods or by the entities is a value
       for each time period or entity. Earlier versions raised "weights must have
       the same number of observations as dependent", or an ``AmbiguityError`` if
       the numbers of time periods and entities were equal. A DataFrame with one
       column and the default index is read as an array with a column is. It raised
       an error from pandas or NumPy if ``dependent`` was balanced, and "Weights do
       not have a supported shape" if it was not. A DataFrame with one column and
       an index that is not the default index is read as a Series with that index
       is, so that its labels decide. It raised "Weights do not have a supported
       shape".
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ):
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)
        if self._constant:
            raise ValueError(
                "Constants are not allowed in first difference regressions."
            )
        if self.dependent.nobs < 2:
            raise ValueError("Panel must have at least 2 time periods")

    def _setup_clusters(
        self,
        cov_config: Mapping[
            str,
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData,
        ],
    ) -> dict[
        str,
        bool | float | str | linearmodels.typing.data.IntArray | DataFrame | PanelData,
    ]:
        cov_config_upd = dict(cov_config).copy()
        cluster_types = ("clusters", "cluster_entity")
        common = set(cov_config.keys()).intersection(cluster_types)
        if not common:
            return cov_config_upd

        clusters = cov_config.get("clusters", None)
        clusters_frame: DataFrame | None = None
        if clusters is not None:
            cluster_data = cast(
                "Union[linearmodels.typing.data.IntArray, DataFrame, PanelData]",
                clusters,
            )
            clusters_panel = self.reformat_clusters(cluster_data)
            fd = clusters_panel.first_difference()
            fd_array = fd.values2d
            if np.any(fd_array.flat[np.isfinite(fd_array.flat)] != 0):
                raise ValueError(
                    "clusters must be identical for values used "
                    "to compute the first difference"
                )
            clusters_frame = clusters_panel.dataframe.copy()

        cluster_entity = cov_config_upd.pop("cluster_entity", False)
        if cluster_entity:
            group_ids = self.dependent.entity_ids.squeeze()
            name = "cov.cluster.entity"
            group_ids_s: Series[int] = Series(
                group_ids, index=self.dependent.index, name=name
            )
            if clusters_frame is not None:
                clusters_frame[name] = group_ids_s
            else:
                clusters_frame = DataFrame(group_ids_s)
        assert clusters_frame is not None
        cluster_data = PanelData(clusters_frame)
        values = cluster_data.values3d[:, 1:]
        cluster_frame = panel_to_frame(
            values,
            cluster_data.panel.items,
            cluster_data.panel.major_axis[1:],
            cluster_data.panel.minor_axis,
            True,
        )
        cluster_frame = PanelData(cluster_frame).dataframe
        cluster_frame = cluster_frame.loc[self.dependent.first_difference().index]
        cluster_frame = cluster_frame.astype(np.int64)

        cov_config_upd["clusters"] = (
            cluster_frame.values if cluster_frame is not None else None
        )

        return cov_config_upd

    def fit(
        self,
        *,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: (
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData
        ),
    ) -> PanelResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        cov_type : str, optional
            Name of covariance estimator. The default is "unadjusted". The names
            are not case sensitive. Supported covariance estimators are (see
            Notes):

            * "unadjusted", "conventional", "homoskedastic" - Homoskedastic
              estimator
            * "robust", "heteroskedastic" - Heteroskedastic estimator
            * "clustered" - Clustered estimator
            * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay estimator
            * "ac", "autocorrelated" - Autocorrelation robust estimator

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True. See Notes.
        **cov_config : bool, float, str, array_like, DataFrame or PanelData
            Additional covariance-specific options. The options that each
            covariance estimator supports are listed in Notes. A key that the
            estimator does not support raises a ``ValueError``.

        Returns
        -------
        PanelResults
            Estimation results.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` is not one of the supported names, or if it is
            "kernel" or "ac" and ``kernel`` is not one of the supported kernels.
            It is both a ``ValueError`` and a ``KeyError``.
        ValueError
            If a key of ``cov_config`` is not supported by the covariance
            estimator (this is checked before the model is estimated), if
            ``clusters`` does not have the shape of the data, if ``clusters`` has
            a missing value for an observation that the model uses, if
            ``bandwidth`` is not below the number of time periods, or if the model
            has no residual degrees of freedom.
        TypeError
            If the value of a key of ``cov_config`` does not have the type that the
            key needs, e.g., if ``kernel`` is not a str.

        Notes
        -----
        Five covariance estimators are supported:

        * "unadjusted", "conventional", "homoskedastic" - Assume the residuals are
          homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered" - One- or two-way clustering. The configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables. This can be
            a pandas object or PanelData with the index of ``dependent``, a 1-d
            array or list with a value for each row of ``dependent`` in the
            order of its rows, or an array or list with the layout of
            ``dependent`` (time by entity, or variable by time by entity).
            An array or list with the layout, or a DataFrame with the time
            periods as its index and the entities as its columns, has a value
            for each time period and entity. It is placed on the rows of
            ``dependent``, which need not be balanced: the values of the
            combinations that ``dependent`` does not have are not used. See
            :ref:`panel-input-alignment`.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables.
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters. The default is False.
          * ``group_debias`` - Boolean flag indicating whether to apply an
            adjustment for the number of clusters. It multiplies the
            covariance by g / (g - 1) * (n - 1) / n, where g is the number
            of distinct clusters and n is the number of observations. With
            two-way clustering, it is applied to each of the three one-way
            covariances that are combined. The default is False.

        * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay HAC estimator. The
          configuration options are:

          * ``kernel`` - One of the supported kernels: "bartlett", "newey-west",
            "parzen", "gallant", "qs", "quadratic-spectral" or "andrews". The names
            are not case sensitive. The default is "newey-west", which is
            Bartlett's kernel.
          * ``bandwidth`` - Bandwidth to use when computing the kernel. If
            not provided, a naive default is used. It must be below the number
            of time periods of the differences, which is one less than the
            number of time periods, and a ``ValueError`` is raised otherwise.

        * "ac", "autocorrelated" - Autocorrelation robust estimator, which allows
          for autocorrelation of the errors of an entity and not for correlation
          between entities. The configuration options are ``kernel`` and
          ``bandwidth``, as for "kernel". The lags of an entity are the positions of
          its differences, not calendar periods, and the bandwidth can be larger
          than the number of differences of an entity. See
          :class:`~linearmodels.panel.covariance.ACCovariance`.

        When using a clustered covariance estimator, all cluster ids must be
        identical within a first difference.  In most scenarios, this requires
        ids to be identical within an entity.

        ``debiased=True``, which is the default, scales the covariance by
        n / (n - k), where k is the number of regressors, and uses the t and
        F distributions for inference. See :ref:`panel-small-sample`.

        ``extra_df`` can be used with any of the covariance estimators. It is the
        number of degrees of freedom that the model uses beyond the regressors,
        which the covariance estimator always accounts for, whether or not
        ``debiased`` is True. The default is 0.

        .. versionchanged:: 7.1

           A ``kernel`` that is not one of the supported names raises an
           ``UnknownOptionError`` when the model is fit, for "kernel" and "ac".
           Earlier versions raised it when the covariance was first used, for
           example when ``std_errors`` of the results was read, so that ``fit``
           returned results that raised when the covariance was used.

        .. versionchanged:: 7.1

           ``clusters`` with the layout of ``dependent`` (time by entity, or
           variable by time by entity) can be used if ``dependent`` is not
           balanced. It raised ``IndexError: Boolean index has wrong length``.
           It is placed on the rows of ``dependent`` by time period and entity.
           Earlier versions paired it by position with the rows, which is right
           only if the rows are sorted by entity and then time: if they were sorted
           by time and then entity, an array was attached to the wrong
           observations without an error, and a pandas object with labels was
           rejected.

        .. versionchanged:: 7.1

           A ``ValueError`` is raised if ``clusters`` has a missing value for an
           observation that the model uses, as in the IV and system models. Earlier
           versions ignored a missing value in the first period of an entity, and
           raised other errors for the others, such as the error of pandas about
           the conversion of NaN to an integer. The clusters of the observations
           that are dropped because a value of ``dependent``, ``exog`` or
           ``weights`` is missing are not used, and can be missing.

        Examples
        --------
        >>> from linearmodels import FirstDifferenceOLS
        >>> from linearmodels.datasets import wage_panel
        >>> data = wage_panel.load().set_index(["nr", "year"])
        >>> y = data["lwage"]
        >>> x = data[["married", "union"]]
        >>> mod = FirstDifferenceOLS(y, x)
        >>> robust = mod.fit(cov_type="robust")
        >>> clustered = mod.fit(cov_type="clustered", cluster_entity=True)
        """
        cov_type = lower_option(cov_type)
        check_covariance_kwargs(
            self._cov_estimators,
            cov_type,
            cov_config,
            cluster_options=("cluster_entity",),
        )
        cov_config = self._setup_clusters(cov_config)
        y_fd = self.dependent.first_difference()
        time_ids = y_fd.time_ids
        entity_ids = y_fd.entity_ids
        index = y_fd.index
        y = cast("linearmodels.typing.data.Float64Array", y_fd.values2d)
        x = cast(
            "linearmodels.typing.data.Float64Array",
            self.exog.first_difference().values2d,
        )

        if np.all(self.weights.values2d == 1.0):
            w = root_w = np.ones_like(y)
        else:
            w = cast(
                "linearmodels.typing.data.Float64Array", 1.0 / self.weights.values3d
            )
            w = w[:, :-1] + w[:, 1:]
            w = cast("linearmodels.typing.data.Float64Array", 1.0 / w)
            w_frame = panel_to_frame(
                w,
                self.weights.panel.items,
                self.weights.panel.major_axis[1:],
                self.weights.panel.minor_axis,
                True,
            )
            w_frame = w_frame.reindex(self.weights.index).dropna(how="any")
            index = cast("MultiIndex", w_frame.index)
            w = np.require(w_frame, requirements="W")

            w /= w.mean()
            root_w = cast("linearmodels.typing.data.Float64Array", np.sqrt(w))

        wx = root_w * x
        wy = root_w * y
        params = _lstsq(wx, wy, rcond=None)[0]
        df_resid = y.shape[0] - x.shape[1]
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.pop("extra_df")
            assert isinstance(_extra_df, (str, int))
            extra_df = int(_extra_df)
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            wx,
            params,
            entity_ids,
            time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = wy - wx @ params
        fitted = DataFrame(
            self.exog.values2d @ params, self.dependent.index, ["fitted_values"]
        )
        idiosyncratic = DataFrame(
            self.dependent.values2d - fitted.values,
            self.dependent.index,
            ["idiosyncratic"],
        )
        effects = DataFrame(
            np.full_like(np.asarray(fitted), np.nan),
            self.dependent.index,
            ["estimated_effects"],
        )
        eps = y - x @ params

        residual_ss = float(np.squeeze(weps.T @ weps))
        total_ss = float(np.squeeze(w.T @ (y**2)))
        # The size of the dependent variable, before it is differenced
        levels = self.dependent.values2d
        y_ss = float(np.squeeze(self.weights.values2d.T @ (levels**2)))
        r2 = r_squared(residual_ss, total_ss, y_ss, y.shape[0])

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, weights=w
        )
        res.update(
            {
                "df_resid": df_resid,
                "df_model": x.shape[1],
                "nobs": y.shape[0],
                "residual_ss": residual_ss,
                "total_ss": total_ss,
                "r2": r2,
                "resids": eps,
                "wresids": weps,
                "index": index,
                "fitted": fitted,
                "effects": effects,
                "idiosyncratic": idiosyncratic,
            }
        )

        return PanelResults(res)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> FirstDifferenceOLS:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules.
        data : PanelData or DataFrame
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time. A
            list or a tuple raises a ``TypeError`` that names ``data``.
        weights : array_like, optional
            Weights to use in estimation. Assumes the variance of the residual is
            proportional to the inverse of the weight, so that the residual times
            the square root of the weight is homoskedastic.
            The weights must be strictly positive and finite. They are rescaled to have
            a mean of 1. A NaN weight is missing, and its observation is dropped.
        check_rank : bool, optional
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank. The default is True.

        Returns
        -------
        FirstDifferenceOLS
            Model specified using the formula.

        Notes
        -----
        Unlike standard  formula syntax, it is necessary to explicitly include
        a constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import FirstDifferenceOLS
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = FirstDifferenceOLS.from_formula("y ~ x1", panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data, context=capture_context(1))
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights, check_rank=check_rank)
        mod.formula = formula
        return mod


class RandomEffects(_PanelModelBase):
    r"""
    One-way Random Effects model for panel data.

    Parameters
    ----------
    dependent : PanelData, DataFrame, Series, ndarray or DataArray
        Dependent (left-hand-side) variable (time by entity). A DataFrame or
        Series with a 2-level MultiIndex (entity, then time), a DataFrame with
        time as the index and entities as the columns, a 2-d (time by entity)
        NumPy array or DataArray, or a PanelData. A list, a tuple or any other
        container raises a ``TypeError`` that names ``dependent``, since the
        entities and time periods need labels or a layout.
    exog : PanelData, DataFrame, Series, ndarray or DataArray
        Exogenous or right-hand-side variables (variable by time by entity). The
        containers are those of ``dependent``, where a DataFrame has a column for
        each variable and an array or DataArray with several variables has 3
        dimensions (variable by time by entity). A list, a tuple or any other
        container raises a ``TypeError`` that names ``exog``.
    weights : array_like, optional
        Weights to use in estimation. Assumes the variance of the residual is
        proportional to the inverse of the weight, so that the residual times
        the square root of the weight is homoskedastic.
        Weights can be given for each observation, for each time period,
        for each entity or for each combination of time period and entity.
        The weights must be strictly positive and finite. They are rescaled to have a
        mean of 1. A NaN weight is missing, and its observation is dropped.
        Weights with labels must use the same labels, in the same order,
        as dependent or, if dependent has no labels, as exog. See
        :ref:`panel-input-alignment`.
        Dependent need not be balanced when the weights are given for each time
        period, entity or combination: the weights of the combinations that
        dependent does not have are not used.
        A 1-d array has a value for each row of dependent, in the order of the
        rows, if it has as many values as dependent has rows. It has a value for
        each time period if it has as many values as there are time periods, and
        a value for each entity if it has as many as there are entities. Weights
        with labels (a pandas object with an index that is not the default index)
        are read by their labels, so that a Series indexed by the time periods has
        a value for each time period, and one indexed by the entities has a value
        for each entity. Weights without labels (an array, a list, a tuple, or a
        Series or a DataFrame with one column that has the default index) are read
        by their number of values. If dependent is not balanced and has as many
        rows as time periods or entities, they have a value for each row. An
        ``AmbiguityError`` is raised if the numbers of time periods and entities
        are equal and the weights have that many values, unless they have no
        labels and a value for each row. Any other number of values raises a
        ``ValueError``.
    check_rank : bool, optional
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank. The default is True.

    Raises
    ------
    TypeError
        If `dependent` or `exog` is a container that is not supported, e.g., a list
        or a tuple, or if they have complex values or values that cannot be
        converted to float64.
    ValueError
        If `dependent`, `exog` or `weights` have different shapes or labels, if a
        combination of an entity and a time period appears more than once, if the
        weights are not strictly positive and finite or do not have a supported
        shape, or if `exog` does not have full column rank and `check_rank` is
        True.
    AmbiguityError
        If the numbers of time periods and entities are equal and 1-d weights have
        that many values, so that they could be a value for each time period or
        for each entity.

    Notes
    -----
    The model is given by

    .. math::

        y_{it} = \beta^{\prime}x_{it} + u_i + \epsilon_{it}

    where :math:`u_i` is a shock that is independent of :math:`x_{it}` but
    common to all entities i.

    .. versionchanged:: 7.1

       ``weights`` for each time period, each entity or each time period and
       entity can be used if ``dependent`` is not balanced. Earlier versions
       raised "weights must have the same number of observations as dependent".

    .. versionchanged:: 7.1

       ``weights`` that are a 1-d array with a value for each row of ``dependent``
       can be used if ``dependent`` is not balanced. Earlier versions raised
       "Weights do not have a supported shape". An array with a value for each
       combination of a time period and an entity is not a value for each row if
       ``dependent`` is not balanced. It raised an error from pandas or NumPy, e.g.,
       "could not broadcast input array from shape (30,1) into shape (27,1)", and
       raises "Weights do not have a supported shape" now, which names the numbers
       of values that are accepted. If ``dependent`` is not balanced and has as many
       rows as time periods or entities, weights without labels (an array, or a
       Series with the default index) with that many values are a value for each
       row, and a Series indexed by the time periods or by the entities is a value
       for each time period or entity. Earlier versions raised "weights must have
       the same number of observations as dependent", or an ``AmbiguityError`` if
       the numbers of time periods and entities were equal. A DataFrame with one
       column and the default index is read as an array with a column is. It raised
       an error from pandas or NumPy if ``dependent`` was balanced, and "Weights do
       not have a supported shape" if it was not. A DataFrame with one column and
       an index that is not the default index is read as a Series with that index
       is, so that its labels decide. It raised "Weights do not have a supported
       shape".
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> None:
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> RandomEffects:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules.
        data : PanelData or DataFrame
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time. A
            list or a tuple raises a ``TypeError`` that names ``data``.
        weights : array_like, optional
            Weights to use in estimation. Assumes the variance of the residual is
            proportional to the inverse of the weight, so that the residual times
            the square root of the weight is homoskedastic.
            The weights must be strictly positive and finite. They are rescaled to have
            a mean of 1. A NaN weight is missing, and its observation is dropped.
        check_rank : bool, optional
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank. The default is True.

        Returns
        -------
        RandomEffects
            Model specified using the formula.

        Notes
        -----
        Unlike standard  formula syntax, it is necessary to explicitly include
        a constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import RandomEffects
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = RandomEffects.from_formula("y ~ 1 + x1", panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data, context=capture_context(1))
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights, check_rank=check_rank)
        mod.formula = formula
        return mod

    def fit(
        self,
        *,
        small_sample: bool = False,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: (
            bool
            | float
            | str
            | linearmodels.typing.data.IntArray
            | DataFrame
            | PanelData
        ),
    ) -> RandomEffectsResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        small_sample : bool, optional
            Apply a small-sample correction to the estimate of the variance of
            the random effect. The default is False.
        cov_type : str, optional
            Name of covariance estimator. The default is "unadjusted". The names
            are not case sensitive. Supported covariance estimators are (see
            Notes):

            * "unadjusted", "conventional", "homoskedastic" - Homoskedastic
              estimator
            * "robust", "heteroskedastic" - Heteroskedastic estimator
            * "clustered" - Clustered estimator
            * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay estimator
            * "ac", "autocorrelated" - Autocorrelation robust estimator

        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True. See Notes.
        **cov_config : bool, float, str, array_like, DataFrame or PanelData
            Additional covariance-specific options. The options that each
            covariance estimator supports are listed in Notes. A key that the
            estimator does not support raises a ``ValueError``.

        Returns
        -------
        RandomEffectsResults
            Estimation results.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` is not one of the supported names, or if it is
            "kernel" or "ac" and ``kernel`` is not one of the supported kernels.
            It is both a ``ValueError`` and a ``KeyError``.
        ValueError
            If a key of ``cov_config`` is not supported by the covariance
            estimator (this is checked before the model is estimated), if
            ``clusters`` does not have the shape of the data, if ``clusters`` has
            a missing value for an observation that the model uses, if
            ``bandwidth`` is not below the number of time periods, if the model
            has no residual degrees of freedom, or if the within or the between
            regression has no residual degrees of freedom, so that the variance of
            the idiosyncratic error or of the effects cannot be estimated.
        TypeError
            If the value of a key of ``cov_config`` does not have the type that the
            key needs, e.g., if ``kernel`` is not a str.

        Notes
        -----
        Five covariance estimators are supported:

        * "unadjusted", "conventional", "homoskedastic" - Assume the residuals are
          homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered" - One- or two-way clustering. The configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables. This can be
            a pandas object or PanelData with the index of ``dependent``, a 1-d
            array or list with a value for each row of ``dependent`` in the
            order of its rows, or an array or list with the layout of
            ``dependent`` (time by entity, or variable by time by entity).
            An array or list with the layout, or a DataFrame with the time
            periods as its index and the entities as its columns, has a value
            for each time period and entity. It is placed on the rows of
            ``dependent``, which need not be balanced: the values of the
            combinations that ``dependent`` does not have are not used. See
            :ref:`panel-input-alignment`.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables.
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters. The default is False.
          * ``cluster_time`` - Boolean flag indicating to use time clusters. The
            default is False.
          * ``group_debias`` - Boolean flag indicating whether to apply an
            adjustment for the number of clusters. It multiplies the
            covariance by g / (g - 1) * (n - 1) / n, where g is the number
            of distinct clusters and n is the number of observations. With
            two-way clustering, it is applied to each of the three one-way
            covariances that are combined. The default is False.

        * "kernel", "driscoll-kraay", "dk" - Driscoll-Kraay HAC estimator. The
          configuration options are:

          * ``kernel`` - One of the supported kernels: "bartlett", "newey-west",
            "parzen", "gallant", "qs", "quadratic-spectral" or "andrews". The names
            are not case sensitive. The default is "newey-west", which is
            Bartlett's kernel.
          * ``bandwidth`` - Bandwidth to use when computing the kernel. If
            not provided, a naive default is used. It must be below the number
            of time periods, and a ``ValueError`` is raised otherwise.

        * "ac", "autocorrelated" - Autocorrelation robust estimator, which allows
          for autocorrelation of the errors of an entity and not for correlation
          between entities. The configuration options are ``kernel`` and
          ``bandwidth``, as for "kernel". The lags of an entity are the positions of
          its observations, not calendar periods, and the bandwidth can be larger
          than the number of observations of an entity. See
          :class:`~linearmodels.panel.covariance.ACCovariance`.

        ``debiased=True``, which is the default, scales the covariance by
        n / (n - k), where k is the number of regressors, and uses the t and
        F distributions for inference.

        ``extra_df`` can be used with any of the covariance estimators. It is the
        number of degrees of freedom that the model uses beyond the regressors,
        which the covariance estimator always accounts for, whether or not
        ``debiased`` is True. The default is 0.

        ``debiased=True`` and ``group_debias=True`` give the clustered
        covariance of Stata's ``xtreg, re vce(cluster ...)``.
        See :ref:`panel-small-sample`.

        The variance of the idiosyncratic error is the residual variance of the
        within regression, and the variance of the effects is estimated from the
        residuals of the between regression. The degrees of freedom of the
        variances are the numbers of coefficients that the regressions estimate. A
        regressor that is constant within entities (the constant, or years of
        education in a wage panel) or that is collinear with the others once the
        means of the entities are removed (experience, if the model has year
        dummies) is not in the within regression, and a regressor that is collinear
        with the others in the means of the entities (the year dummies and the
        constant in a balanced panel) is not in the between regression. A
        ``ValueError`` is raised if either regression has no residual degrees of
        freedom, since the variances cannot be estimated.

        .. versionchanged:: 7.1

           The degrees of freedom of the variances count the coefficients that the
           regressions estimate. Earlier versions counted every column of ``exog``
           (less one for the constant in the within regression), so that the
           variances, theta and the estimates were slightly different for models with
           time-invariant or collinear regressors, and for models without a constant.
           With ``small_sample`` and an unbalanced panel, the residuals of the between
           regression that the estimator of Baltagi and Chang (1994) uses are those of
           the regression that weights the entities by their numbers of observations.
           Earlier versions used the residuals of the unweighted regression of the
           means, which gave a different variance of the effects. Models that do not
           have residual degrees of freedom in the regressions raise a
           ``ValueError``. Earlier versions raised an error from NumPy or SciPy,
           or returned variances that were not defined.

        .. versionchanged:: 7.1

           A ``kernel`` that is not one of the supported names raises an
           ``UnknownOptionError`` when the model is fit, for "kernel" and "ac".
           Earlier versions raised it when the covariance was first used, for
           example when ``std_errors`` of the results was read, so that ``fit``
           returned results that raised when the covariance was used.

        .. versionchanged:: 7.1

           ``clusters`` with the layout of ``dependent`` (time by entity, or
           variable by time by entity) can be used if ``dependent`` is not
           balanced. It raised ``IndexError: Boolean index has wrong length``.
           It is placed on the rows of ``dependent`` by time period and entity.
           Earlier versions paired it by position with the rows, which is right
           only if the rows are sorted by entity and then time: if they were sorted
           by time and then entity, an array was attached to the wrong
           observations without an error, and a pandas object with labels was
           rejected.

        .. versionchanged:: 7.1

           A ``ValueError`` is raised if ``clusters`` has a missing value for an
           observation that the model uses, as in the IV and system models. Earlier
           versions made the observations with a missing value a cluster of their
           own, so that the standard errors changed without an error. The clusters
           of the observations that are dropped because a value of ``dependent``,
           ``exog`` or ``weights`` is missing are not used, and can be missing.

        Examples
        --------
        >>> from linearmodels import RandomEffects
        >>> from linearmodels.datasets import wage_panel
        >>> data = wage_panel.load().set_index(["nr", "year"])
        >>> y = data["lwage"]
        >>> x = data[["married", "union"]]
        >>> mod = RandomEffects(y, x)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)
        """
        cov_type = lower_option(cov_type)
        check_covariance_kwargs(self._cov_estimators, cov_type, cov_config)
        cov_config = self._setup_clusters(cov_config)
        w = self.weights.values2d
        root_w = cast("linearmodels.typing.data.Float64Array", np.sqrt(w))
        demeaned_dep = self.dependent.demean("entity", weights=self.weights)
        demeaned_exog = self.exog.demean("entity", weights=self.weights)
        assert isinstance(demeaned_dep, PanelData)
        assert isinstance(demeaned_exog, PanelData)
        y = demeaned_dep.values2d
        x = demeaned_exog.values2d
        # The regressors that do not vary within entities (a constant, a time-invariant
        # regressor) or that are collinear once the means are removed have no
        # coefficient in the within regression, and do not use a degree of freedom.
        # Removing the means leaves rounding error in a regressor that does not vary,
        # which is large if its units are large, so the rank is that of the regressors
        # in the units that they had before the means were removed
        x_norm = np.sqrt(((root_w * self.exog.values2d) ** 2).sum(0))
        x_norm[x_norm == 0.0] = 1.0
        rank_within = _lstsq(x / x_norm, y, rcond=None)[2]
        if self.has_constant:
            w_sum = w.sum()
            y_gm = (w * self.dependent.values2d).sum(0) / w_sum
            x_gm = (w * self.exog.values2d).sum(0) / w_sum
            y += root_w * y_gm
            x += root_w * x_gm
        params = _lstsq(x, y, rcond=None)[0]
        weps = y - x @ params

        wybar = self.dependent.mean("entity", weights=self.weights)
        wxbar = self.exog.mean("entity", weights=self.weights)
        params, _, rank_between, _ = _lstsq(
            np.asarray(wxbar), np.asarray(wybar), rcond=None
        )
        wu = np.asarray(wybar) - np.asarray(wxbar) @ params

        nobs = weps.shape[0]
        neffects = wu.shape[0]
        df_within = nobs - neffects - rank_within
        df_between = neffects - rank_between
        if df_within <= 0:
            raise ValueError(
                "The variance of the idiosyncratic error cannot be estimated since "
                "the within regression has no residual degrees of freedom. The model "
                f"has {nobs} observations of {neffects} entities, and the regressors "
                f"that vary within entities have a rank of {rank_within}, so that "
                f"at least {neffects + rank_within + 1} observations are needed."
            )
        if df_between <= 0:
            raise ValueError(
                "The variance of the effects cannot be estimated since the between "
                "regression has no residual degrees of freedom. The model has "
                f"{neffects} entities, and the means of the regressors have a rank "
                f"of {rank_between}, so that at least {rank_between + 1} entities "
                "are needed."
            )
        sigma2_e = float(np.squeeze(weps.T @ weps)) / df_within
        ssr = float(np.squeeze(wu.T @ wu))
        t = np.asarray(self.dependent.count("entity"))
        unbalanced = np.ptp(t) != 0
        if small_sample and unbalanced:
            # The between regression of the estimator of Baltagi and Chang (1994) is
            # the regression of the data replaced by the means of the entities. It
            # weights the means by the numbers of observations, as the trace below
            # does, and the residuals of the regression of the means do not
            root_t = np.sqrt(t)
            xbar, ybar = np.asarray(wxbar), np.asarray(wybar)
            params_t = _lstsq(root_t * xbar, root_t * ybar, rcond=None)[0]
            wu_t = ybar - xbar @ params_t
            ssr = float(np.squeeze((t * wu_t).T @ wu_t))
            wx_df = cast("DataFrame", root_w * self.exog.dataframe)
            means = wx_df.groupby(level=0).transform("mean").values
            denom = means.T @ means
            sums = wx_df.groupby(level=0).sum().values
            num = sums.T @ sums
            # The means of the regressors can be collinear, e.g., the dummy variables
            # of two periods that are observed for every entity have the same mean
            # in all entities. The pseudo inverse uses the regressors that are not
            # collinear
            tr = np.trace(np.linalg.pinv(denom, hermitian=True) @ num)
            sigma2_u = max(0, (ssr - df_between * sigma2_e) / (nobs - tr))
        else:
            t_bar = neffects / ((1.0 / t).sum())
            sigma2_u = max(0, ssr / df_between - sigma2_e / t_bar)
        rho = sigma2_u / (sigma2_u + sigma2_e)

        theta = 1.0 - np.sqrt(sigma2_e / (t * sigma2_u + sigma2_e))
        theta_out = DataFrame(theta, columns=["theta"], index=wybar.index)
        wy: linearmodels.typing.data.Float64Array = np.asarray(
            root_w * self.dependent.values2d, dtype=float
        )
        wx: linearmodels.typing.data.Float64Array = np.asarray(
            root_w * self.exog.values2d, dtype=float
        )
        # The entity means are in the order in which the entities appear. Rows are
        # selected by position since dependent and exog need not have the same labels
        entity_loc = factorize(self.dependent.index.get_level_values(0))[0]
        wybar_rows = np.asarray(theta * wybar)[entity_loc]
        wxbar_rows = np.asarray(theta * wxbar)[entity_loc]
        wy -= root_w * wybar_rows
        wx -= root_w * wxbar_rows
        params = _lstsq(wx, wy, rcond=None)[0]

        df_resid = wy.shape[0] - wx.shape[1]
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.pop("extra_df")
            assert isinstance(_extra_df, (str, int))
            extra_df = int(_extra_df)
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            np.asarray(wx),
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = wy - wx @ params
        eps = weps / root_w
        index = self.dependent.index
        fitted = DataFrame(self.exog.values2d @ params, index, ["fitted_values"])
        effects = DataFrame(
            self.dependent.values2d - np.asarray(fitted) - eps,
            index,
            ["estimated_effects"],
        )
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        residual_ss = float(np.squeeze(weps.T @ weps))
        wmu: float | linearmodels.typing.data.Float64Array = 0.0
        if self.has_constant:
            wmu = root_w * _lstsq(root_w, wy, rcond=None)[0]
        wy_demeaned = wy - wmu
        total_ss = float(np.squeeze(wy_demeaned.T @ wy_demeaned))
        # The size of the dependent variable, before the means of the entities are removed
        y_ss = float(np.squeeze(w.T @ (self.dependent.values2d**2)))
        r2 = r_squared(residual_ss, total_ss, y_ss, wy.shape[0])

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, weights=w
        )
        res.update(
            {
                "df_resid": df_resid,
                "df_model": x.shape[1],
                "nobs": y.shape[0],
                "residual_ss": residual_ss,
                "total_ss": total_ss,
                "r2": r2,
                "resids": eps,
                "wresids": weps,
                "index": index,
                "sigma2_eps": sigma2_e,
                "sigma2_effects": sigma2_u,
                "rho": rho,
                "theta": theta_out,
                "fitted": fitted,
                "effects": effects,
                "idiosyncratic": idiosyncratic,
            }
        )

        return RandomEffectsResults(res)


class FamaMacBeth(_PanelModelBase):
    r"""
    Fama-MacBeth estimator for panel data.

    Parameters
    ----------
    dependent : PanelData, DataFrame, Series, ndarray or DataArray
        Dependent (left-hand-side) variable (time by entity). A DataFrame or
        Series with a 2-level MultiIndex (entity, then time), a DataFrame with
        time as the index and entities as the columns, a 2-d (time by entity)
        NumPy array or DataArray, or a PanelData. A list, a tuple or any other
        container raises a ``TypeError`` that names ``dependent``, since the
        entities and time periods need labels or a layout.
    exog : PanelData, DataFrame, Series, ndarray or DataArray
        Exogenous or right-hand-side variables (variable by time by entity). The
        containers are those of ``dependent``, where a DataFrame has a column for
        each variable and an array or DataArray with several variables has 3
        dimensions (variable by time by entity). A list, a tuple or any other
        container raises a ``TypeError`` that names ``exog``.
    weights : array_like, optional
        Weights to use in estimation. Assumes the variance of the residual is
        proportional to the inverse of the weight, so that the residual times
        the square root of the weight is homoskedastic.
        Weights can be given for each observation, for each time period,
        for each entity or for each combination of time period and entity.
        The weights must be strictly positive and finite. They are rescaled to have a
        mean of 1. A NaN weight is missing, and its observation is dropped.
        Weights with labels must use the same labels, in the same order,
        as dependent or, if dependent has no labels, as exog. See
        :ref:`panel-input-alignment`.
        Dependent need not be balanced when the weights are given for each time
        period, entity or combination: the weights of the combinations that
        dependent does not have are not used.
        A 1-d array has a value for each row of dependent, in the order of the
        rows, if it has as many values as dependent has rows. It has a value for
        each time period if it has as many values as there are time periods, and
        a value for each entity if it has as many as there are entities. Weights
        with labels (a pandas object with an index that is not the default index)
        are read by their labels, so that a Series indexed by the time periods has
        a value for each time period, and one indexed by the entities has a value
        for each entity. Weights without labels (an array, a list, a tuple, or a
        Series or a DataFrame with one column that has the default index) are read
        by their number of values. If dependent is not balanced and has as many
        rows as time periods or entities, they have a value for each row. An
        ``AmbiguityError`` is raised if the numbers of time periods and entities
        are equal and the weights have that many values, unless they have no
        labels and a value for each row. Any other number of values raises a
        ``ValueError``.
    check_rank : bool, optional
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank. The default is True.

    Raises
    ------
    TypeError
        If `dependent` or `exog` is a container that is not supported, e.g., a list
        or a tuple, or if they have complex values or values that cannot be
        converted to float64.
    ValueError
        If `dependent`, `exog` or `weights` have different shapes or labels, if a
        combination of an entity and a time period appears more than once, if the
        weights are not strictly positive and finite or do not have a supported
        shape, or if `exog` does not have full column rank and `check_rank` is
        True. It is also raised if none of the cross-sectional regressions can be
        estimated.
    AmbiguityError
        If the numbers of time periods and entities are equal and 1-d weights have
        that many values, so that they could be a value for each time period or
        for each entity.

    Notes
    -----
    The model is given by

    .. math::

        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}

    The Fama-MacBeth estimator is computed by performing T regressions, one
    for each time period using all available entity observations.  Denote the
    estimate of the model parameters as :math:`\hat{\beta}_t`.  The reported
    estimator is then

    .. math::

        \hat{\beta} = T^{-1}\sum_{t=1}^T \hat{\beta}_t

    While the model does not explicitly include time-effects, the
    implementation based on regressing all observation in a single
    time period is "as-if" time effects are included.

    Each period needs at least as many entities as regressors, and the
    regressors must vary across the entities of the period, so that they are
    not collinear: a regressor that depends only on time is collinear with the
    constant. Periods whose cross-section is rank deficient are dropped with a
    warning, and T counts only the periods that are used. It is an error if no
    period can be used.

    Parameter inference is made using the set of T parameter estimates with
    either the standard covariance estimator or a kernel-based covariance,
    depending on ``cov_type``. Since the parameters are the averages of the T
    cross-sectional estimates, the t-statistics are compared with a t
    distribution with T - 1 degrees of freedom, which is the convention of
    Fama and MacBeth [1]_. T is the number of periods whose cross-sectional
    regression is estimated. The p-values and confidence intervals use this
    distribution if ``debiased`` is True, which is the default, and the normal
    distribution otherwise.

    The joint test that the k parameters other than the constant are zero,
    ``f_statistic_robust`` of the results, is the Hotelling T-squared test
    [2]_. The Wald statistic W of the k parameters is multiplied by
    (T - k) / (k (T - 1)) and compared with an F distribution with k and T - k
    degrees of freedom if ``debiased`` is True, and W is compared with a
    chi-squared distribution with k degrees of freedom otherwise. The F test is
    exact if the T estimates are independent draws from the same normal
    distribution and ``cov_type`` is not "kernel". With the kernel covariance
    it is only an approximation, which can reject a true null hypothesis far
    too often when T is small. With one tested parameter it is the squared
    t-statistic, with the p-value of the t distribution with T - 1 degrees of
    freedom. An F distribution with k and T - 1 degrees of freedom, or with
    the degrees of freedom of the pooled regression, rejects a true null
    hypothesis too often when k is larger than 1, see
    :attr:`~linearmodels.panel.results.FamaMacBethResults.f_statistic_robust`.

    .. versionchanged:: 7.1

       ``weights`` for each time period, each entity or each time period and
       entity can be used if ``dependent`` is not balanced. Earlier versions
       raised "weights must have the same number of observations as dependent".

    .. versionchanged:: 7.1

       ``weights`` that are a 1-d array with a value for each row of ``dependent``
       can be used if ``dependent`` is not balanced. Earlier versions raised
       "Weights do not have a supported shape". An array with a value for each
       combination of a time period and an entity is not a value for each row if
       ``dependent`` is not balanced. It raised an error from pandas or NumPy, e.g.,
       "could not broadcast input array from shape (30,1) into shape (27,1)", and
       raises "Weights do not have a supported shape" now, which names the numbers
       of values that are accepted. If ``dependent`` is not balanced and has as many
       rows as time periods or entities, weights without labels (an array, or a
       Series with the default index) with that many values are a value for each
       row, and a Series indexed by the time periods or by the entities is a value
       for each time period or entity. Earlier versions raised "weights must have
       the same number of observations as dependent", or an ``AmbiguityError`` if
       the numbers of time periods and entities were equal. A DataFrame with one
       column and the default index is read as an array with a column is. It raised
       an error from pandas or NumPy if ``dependent`` was balanced, and "Weights do
       not have a supported shape" if it was not. A DataFrame with one column and
       an index that is not the default index is read as a Series with that index
       is, so that its labels decide. It raised "Weights do not have a supported
       shape".

    References
    ----------
    .. [1] Fama, E. F. and MacBeth, J. D. (1973). Risk, return, and
       equilibrium: Empirical tests. Journal of Political Economy, 81(3),
       607-636.
    .. [2] Hotelling, H. (1931). The generalization of Student's ratio.
       The Annals of Mathematical Statistics, 2(3), 360-378.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ):
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)
        self._validate_blocks()

    def _validate_blocks(self) -> None:
        exog = self.exog.dataframe
        root_w = np.sqrt(self.weights.values2d)
        wx_df = DataFrame(
            root_w * self.exog.values2d, index=exog.index, columns=exog.columns
        )

        def validate_block(
            ex: linearmodels.typing.data.Float64Array | DataFrame,
        ) -> bool:
            _ex = np.asarray(ex, dtype=float)

            def _mr(ex: linearmodels.typing.data.Float64Array) -> int:
                """lstsq based matrix_rank"""
                return _lstsq(ex, np.ones(ex.shape[0]))[2]

            return _ex.shape[0] >= _ex.shape[1] and _mr(_ex) == _ex.shape[1]

        valid_blocks = wx_df.groupby(level=1).apply(validate_block)
        if not valid_blocks.any():
            err = (
                "Model cannot be estimated. Fama-MacBeth estimates one "
                "cross-sectional regression for each time period, using the "
                "entities that are observed in the period, and none of these "
                "regressions can be estimated. A cross-section needs at least "
                "as many entities as regressors, and its regressors must not be "
                "collinear, so the regressors must vary across entities within "
                "a period. A regressor that depends only on time is collinear "
                "with the constant, and a panel with a single entity has one "
                "observation in each cross-section. Use more entities, remove "
                "the regressors that do not vary across entities, or, for a "
                "single entity, estimate a time-series regression."
            )
            raise ValueError(err)
        if valid_blocks.sum() < exog.shape[1]:

            warnings.warn(
                "The number of cross-sectional regressions that can be estimated, "
                "{}, is less than the number of parameters in the model. Parameter "
                "inference is not available.".format(valid_blocks.sum()),
                InferenceUnavailableWarning,
                stacklevel=3,
            )
        elif valid_blocks.sum() < valid_blocks.shape[0]:

            warnings.warn(
                "{} of the cross-sectional regressions cannot be estimated due to "
                "deficient rank.".format(valid_blocks.shape[0] - valid_blocks.sum()),
                MissingValueWarning,
                stacklevel=3,
            )

    def fit(
        self,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        bandwidth: float | None = None,
        kernel: str | None = None,
    ) -> FamaMacBethResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        cov_type : {"unadjusted", "homoskedastic", "robust", "heteroskedastic", "kernel"}, optional
            Name of covariance estimator. The default is "unadjusted". The names
            are not case sensitive. "homoskedastic" is an alias of "unadjusted",
            and "heteroskedastic" is an alias of "robust". See Notes.
        debiased : bool, optional
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment, which uses T - 1 instead of T when
            the covariance of the T estimates is computed, and to use a t
            distribution with T - 1 degrees of freedom for inference. If
            False, inference uses the normal distribution. The default is True.
        bandwidth : float, optional
            The bandwidth to use when cov_type is "kernel". It must be below T,
            the number of periods that are used. If not provided, it is
            automatically computed, and is at most T - 1. The automatic bandwidth
            needs T to be at least 4 with the Bartlett kernel, 5 with the Parzen
            kernel and 6 with the quadratic spectral kernel.
        kernel : str, optional
            Name of the kernel. If not provided, "newey-west", which is the
            Bartlett kernel, is used. The names are not case sensitive. It is
            checked for every ``cov_type``. Supported kernels are:

            * "bartlett", "newey-west" - Bartlett's kernel
            * "parzen", "gallant" - Parzen's kernel
            * "qs", "quadratic-spectral", "andrews" - Quadratic spectral kernel

        Returns
        -------
        FamaMacBethResults
            Estimation results.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` is not one of the supported names, or if ``kernel`` is
            not one of the supported names, whether or not ``bandwidth`` is
            given. It is both a ``ValueError`` and a ``KeyError``.
        ValueError
            If ``cov_type`` is "kernel" and ``bandwidth`` is not below T, the
            number of periods that are used. It is also raised if it is "kernel",
            ``bandwidth`` is not provided and T is below the number that the
            automatic bandwidth needs, or the automatic bandwidth is not finite.

        Notes
        -----
        Two covariance estimators are supported:

        * "unadjusted", "homoskedastic", "robust", "heteroskedastic" use the
          standard covariance estimator of the T parameter estimates.
        * "kernel" is a HAC estimator of the covariance of the T parameter
          estimates. The configuration options are ``kernel`` and ``bandwidth``.
          The T estimates have autocovariances up to lag T - 1, so the bandwidth
          must be below T. This holds for every kernel, including the quadratic
          spectral kernel, which is not truncated. The bandwidth is checked when
          the model is fit. If ``bandwidth`` is None it is computed from the T
          estimates by
          :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`, when the
          model is fit as well. The rule needs T to be at least 4 with the
          Bartlett kernel ("bartlett", "newey-west", the default), 5 with the
          Parzen kernel ("parzen", "gallant") and 6 with the quadratic spectral
          kernel ("qs", "quadratic-spectral", "andrews"). Pass ``bandwidth`` to
          use the estimator with fewer periods.

        The parameters are the averages of the T cross-sectional estimates, and
        T is the number of periods whose cross-sectional regression is
        estimated: periods that cannot be estimated are not counted. If
        ``debiased`` is True, which is the default, the p-values and
        confidence intervals use a t distribution with T - 1 degrees of
        freedom, which is the convention of Fama and MacBeth (1973) and is the
        ``df_resid`` of the results. The standard and the kernel covariance
        estimators use the same distribution. If ``debiased`` is False, they
        use the normal distribution. The estimates and the standard errors
        do not depend on the distribution.

        The joint test of the parameters other than the constant,
        ``f_statistic_robust`` of the results, is the Hotelling T-squared test.
        If ``debiased`` is True, the Wald statistic of the k tested parameters
        is multiplied by (T - k) / (k (T - 1)), and the p-value is computed from
        an F distribution with k and T - k degrees of freedom. If ``debiased``
        is False, the statistic is compared with a chi-squared distribution with
        k degrees of freedom. See
        :attr:`~linearmodels.panel.results.FamaMacBethResults.f_statistic_robust`.

        .. versionchanged:: 7.1

           The p-values and confidence intervals use T - 1 degrees of freedom.
           Earlier versions used nobs - nvar degrees of freedom, those of a
           pooled regression, which is close to the normal distribution when
           there are many observations. ``f_statistic_robust`` uses Hotelling's
           reference. Earlier versions divided the Wald statistic by k and used
           an F distribution with k and nobs - nvar degrees of freedom, which
           rejects a true null hypothesis too often when k is larger than 1.

        .. versionchanged:: 7.1

           A bandwidth that is not below T raises a ``ValueError`` when the model
           is fit. Earlier versions raised "Length of w is larger than the number
           of elements in z" with the Bartlett and Parzen kernels, when the
           covariance was first used. The quadratic spectral kernel, which has no
           truncation, returned a covariance without an error, which was close to
           zero for a large bandwidth.

        .. versionchanged:: 7.1

           The automatic bandwidth raises a ``ValueError`` when the model is fit
           if T is below 4 with the Bartlett kernel, 5 with the Parzen kernel or
           6 with the quadratic spectral kernel. The deviations of the T
           estimates from their average sum to zero, and with so few periods the
           rule divides by the sum of all of their autocovariances. Earlier
           versions raised "cannot convert float infinity to integer" (an
           ``OverflowError``) or "cannot convert float NaN to integer" when the
           covariance was first used, or used a bandwidth of T - 1 because of the
           rounding error of the sum.

        .. versionchanged:: 7.1

           A ``kernel`` that is not one of the supported names raises an
           ``UnknownOptionError`` when the model is fit, whether or not
           ``bandwidth`` is given. Earlier versions raised it when the covariance
           was first used, for example when ``std_errors`` of the results was
           read, so that ``fit`` returned results that raised when the covariance
           was used.

        Examples
        --------
        >>> from linearmodels import FamaMacBeth
        >>> from linearmodels.datasets import wage_panel
        >>> data = wage_panel.load().set_index(["nr", "year"])
        >>> y = data["lwage"]
        >>> x = data[["married", "union"]]
        >>> mod = FamaMacBeth(y, x)
        >>> res = mod.fit(cov_type="kernel", kernel="parzen")
        """
        cov_type = lower_option(cov_type)
        if isinstance(kernel, str):
            lookup_option(KERNEL_LOOKUP, kernel.lower(), "kernel")
        # The retained observations and their weights, which are scaled to have a
        # mean of one once the observations with missing values are dropped
        y = cast("linearmodels.typing.data.Float64Array", self.dependent.values2d)
        x = cast("linearmodels.typing.data.Float64Array", self.exog.values2d)
        root_w = cast(
            "linearmodels.typing.data.Float64Array", np.sqrt(self.weights.values2d)
        )
        wy = cast("linearmodels.typing.data.Float64Array", root_w * y)
        wx = cast("linearmodels.typing.data.Float64Array", root_w * x)

        dep = self.dependent.dataframe
        exog = self.exog.dataframe
        index = self.dependent.index
        wy_df = DataFrame(wy, index=index, columns=dep.columns)
        wx_df = DataFrame(wx, index=exog.index, columns=exog.columns)

        yx = DataFrame(
            np.c_[wy_df.values, wx_df.values],
            columns=list(wy_df.columns) + list(wx_df.columns),
            index=wy_df.index,
        )

        has_constant = self.has_constant
        loadings = self._constant_loadings() if has_constant else None

        def single(z: DataFrame) -> Series:
            exog = z.iloc[:, 1:].values
            cols = [*list(z.columns), "r2", "adv_r2"]
            if exog.shape[0] < exog.shape[1]:
                return Series([np.nan] * (len(z.columns) + 2), index=cols)
            dep = z.iloc[:, :1].values
            params, _, rank, _ = _lstsq(exog, dep)
            nexog = exog.shape[1]
            if rank != nexog:
                return Series([np.nan] * (len(z.columns) + 2), index=cols)
            err = dep - exog @ params
            sse = float(np.squeeze(err.T @ err))
            if loadings is not None:
                # The constant of the weighted data is not a column of ones
                const = exog @ loadings
                dep_demean = dep - const * ((const.T @ dep) / (const.T @ const))
                tss = float(np.squeeze(dep_demean.T @ dep_demean))
            else:
                tss = float(np.squeeze(dep.T @ dep))
            nobs = exog.shape[0]
            r2 = r_squared(sse, tss, float(np.squeeze(dep.T @ dep)), nobs)
            if nobs - nexog > 0 and not np.isnan(r2):
                adj_r2 = 1 - (sse / (nobs - nexog)) / (tss / (nobs - int(has_constant)))
            else:
                adj_r2 = np.nan
            return Series(np.r_[np.nan, params.ravel(), r2, adj_r2], index=cols)

        all_params = yx.groupby(level=1).apply(single)
        avg_r2_values = all_params.iloc[:, -2]
        if np.any(np.isfinite(avg_r2_values)):
            avg_r2 = np.nanmean(avg_r2_values)
        else:
            avg_r2 = np.nan
        avg_adj_r2_values = all_params.iloc[:, -1]
        if np.any(np.isfinite(avg_adj_r2_values)):
            avg_adj_r2 = np.nanmean(avg_adj_r2_values)
        else:
            avg_adj_r2 = np.nan
        all_params = all_params.iloc[:, 1:-2]
        params = np.asarray(all_params.mean(axis=0).values[:, None], dtype=float)

        wy = np.asarray(wy_df)
        wx = np.asarray(wx_df)
        index = self.dependent.index
        fitted = DataFrame(self.exog.values2d @ params, index, ["fitted_values"])
        effects = DataFrame(np.full(fitted.shape, np.nan), index, ["estimated_effects"])
        idiosyncratic = DataFrame(
            self.dependent.values2d - fitted.values, index, ["idiosyncratic"]
        )

        eps = self.dependent.values2d - fitted.values
        weps = wy - wx @ params
        w = self.weights.values2d
        residual_ss = float(np.squeeze(weps.T @ weps))
        e = y
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()
        total_ss = float(np.squeeze(w.T @ (e**2)))
        y_ss = float(np.squeeze(w.T @ (y**2)))
        r2 = r_squared(residual_ss, total_ss, y_ss, y.shape[0])

        if cov_type not in (
            "robust",
            "unadjusted",
            "homoskedastic",
            "heteroskedastic",
            "kernel",
        ):
            supported = (
                "heteroskedastic",
                "homoskedastic",
                "kernel",
                "robust",
                "unadjusted",
            )
            raise unknown_option_error("cov_type", cov_type, supported)

        bandwidth = 0.0 if cov_type != "kernel" else bandwidth
        cov = FamaMacBethCovariance(
            wy,
            wx,
            params,
            all_params,
            debiased=debiased,
            kernel=kernel,
            bandwidth=bandwidth,
        )

        # T is the number of cross-sectional estimates that are averaged, the
        # periods that could be estimated, which are also the periods that the
        # covariance uses. Inference uses t(T - 1)
        nperiod = int(np.isfinite(np.asarray(all_params)).all(axis=1).sum())
        df_resid = nperiod - 1
        # The classical F-statistic is that of the pooled regression
        pooled_df_resid = wy.shape[0] - params.shape[0]
        res = self._postestimation(
            params, cov, debiased, pooled_df_resid, weps, wy, wx, weights=w
        )
        index = self.dependent.index
        res.update(
            {
                "df_resid": df_resid,
                "df_model": x.shape[1],
                "nobs": y.shape[0],
                "residual_ss": residual_ss,
                "total_ss": total_ss,
                "r2": r2,
                "resids": eps,
                "wresids": weps,
                "index": index,
                "fitted": fitted,
                "effects": effects,
                "idiosyncratic": idiosyncratic,
                "all_params": all_params,
                "avg_r2": avg_r2,
                "avg_adj_r2": avg_adj_r2,
            }
        )
        return FamaMacBethResults(res)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: PanelArrayLike | None = None,
        check_rank: bool = True,
    ) -> FamaMacBeth:
        """
        Create a model from a formula.

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules.
        data : PanelData or DataFrame
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time. A
            list or a tuple raises a ``TypeError`` that names ``data``.
        weights : array_like, optional
            Weights to use in estimation. Assumes the variance of the residual is
            proportional to the inverse of the weight, so that the residual times
            the square root of the weight is homoskedastic.
            The weights must be strictly positive and finite. They are rescaled to have
            a mean of 1. A NaN weight is missing, and its observation is dropped.
        check_rank : bool, optional
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank. The default is True.

        Returns
        -------
        FamaMacBeth
            Model specified using the formula.

        Notes
        -----
        Unlike standard  formula syntax, it is necessary to explicitly include
        a constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import FamaMacBeth
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = FamaMacBeth.from_formula("y ~ 1 + x1", panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data, context=capture_context(1))
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights, check_rank=check_rank)
        mod.formula = formula
        return mod
