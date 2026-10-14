"""
Weights that are a 1-d array with a value for each row of dependent

* A 1-d array of weights with a value for each row of dependent raised "Weights do not
  have a supported shape" if dependent was not balanced. The same values as a pandas
  object with the index of dependent, and 1-d arrays of clusters and other effects,
  worked. The array worked if dependent was balanced, since it then had as many values
  as the number of time periods times the number of entities, which is the number of
  rows. An array with that many values raised an error from pandas or NumPy if
  dependent was not balanced.
* A 1-d array is read by its number of values. It has a value for each row of dependent
  if it has as many values as dependent has rows, a value for each time period if it
  has as many as there are time periods, and a value for each entity if it has as many
  as there are entities. If dependent is not balanced, the number of values of a value
  for each row is read first, also if it is the number of time periods or entities.

The values are paired with the rows of dependent by position, before the observations
with missing values are dropped, whatever the order of the rows is.

The reference of the tests is the same model with the weights as a Series with the index
of dependent, whose results must be identical. The weights of the rows are also compared
with the weights that were used to make them, and the estimates with a weighted least
squares fit that does not use the models.
"""

from linearmodels.compat.pandas import ANNUAL_FREQ

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.model import AmbiguityError
from linearmodels.shared.exceptions import MissingValueWarning, SingletonWarning

NENTITY = 12
NPERIOD = 8
LAYOUTS = ["unbalanced", "balanced"]
# The order of the rows of dependent: sorted by entity and then time period, sorted by
# time period and then entity, or in a random order
ORDERS = ["entity-major", "time-major", "shuffled"]
MISSING = ["none", "y", "x"]
# Cells that the unbalanced panel has, with a missing value of the dependent variable
# or of a regressor in the panels with missing values
MISSING_Y = [(1, 3), (2, 1), (4, 5), (9, 2), (10, 6)]
MISSING_X = [(1, 1), (4, 2), (7, 3), (9, 6)]
SHAPE_MESSAGE = "Weights do not have a supported shape"
AMBIGUITY_MESSAGE = (
    "Unable to distinguish nobs from nentity since they are equal. You must use a 2-d "
    "array to avoid ambiguity."
)


def cell_weight(entity, period):
    """The weight of the combination of an entity and a period, by their positions"""
    return 0.5 + ((3 * entity + 5 * period) % 7) / 3


def observed_cells(layout):
    """The combinations of an entity and a period that the panel has"""
    observed = np.ones((NENTITY, NPERIOD), dtype=bool)
    if layout == "unbalanced":
        observed[0, [0, 1]] = False  # enters late
        observed[3, 2] = False  # a gap
        observed[8, 5] = False  # a gap
        observed[[3, 6, 7, 8, 10, 11], 4] = False  # a period with few entities
        observed[5, [0, 1, 2]] = False  # a short entity
        observed[11, :5] = False  # a short entity
        observed[[2, 6], 0] = False  # missing observations in the first period
    return observed


def main_cells(layout, order):
    """The (entity, period) of the rows of the panel of the tests, in order"""
    observed = observed_cells(layout)
    cells = [(e, t) for e in range(NENTITY) for t in range(NPERIOD) if observed[e, t]]
    if order == "time-major":
        cells.sort(key=lambda cell: (cell[1], cell[0]))
    elif order == "shuffled":
        permutation = np.random.default_rng(5).permutation(len(cells))
        cells = [cells[i] for i in permutation]
    return cells


class Panel:
    """
    A panel with its rows in a given order

    Parameters
    ----------
    cells : list of tuple of int
        The (entity, period) of each row, in the order of the rows. The entities and
        the periods are the positions 0, 1, ... of the full panel.
    missing : {"none", "y", "x"}, optional
        Whether observations have a missing value of the dependent variable or of a
        regressor.
    style : {"int", "date"}, optional
        The labels of the periods.
    """

    def __init__(self, cells, missing="none", style="int"):
        self.cells, self.missing = list(cells), missing
        nentity = max(e for e, _ in self.cells) + 1
        nperiod = max(t for _, t in self.cells) + 1
        # Zero padded, so that the order of the labels is that of the positions
        self.entities = [f"firm{i:02d}" for i in range(nentity)]
        if style == "int":
            self.periods = list(range(2001, 2001 + nperiod))
        else:
            dates = pd.date_range("2001-12-31", periods=nperiod, freq=ANNUAL_FREQ)
            self.periods = list(dates)
        self.index = pd.MultiIndex.from_tuples(
            [(self.entities[e], self.periods[t]) for e, t in self.cells],
            names=["entity", "time"],
        )
        rng = np.random.default_rng(20261014)
        x = rng.standard_normal((nentity, nperiod, 2))
        effect = rng.standard_normal(nentity)
        shock = rng.standard_normal((nentity, nperiod))
        y = x @ [1.0, -0.5] + effect[:, None] + shock
        rows = tuple(np.array(self.cells).T)
        self.x = pd.DataFrame(x[rows], index=self.index, columns=["x0", "x1"])
        self.y = pd.Series(y[rows], index=self.index, name="y")
        if missing == "y":
            self.y.iloc[self.positions(MISSING_Y)] = np.nan
        elif missing == "x":
            self.x.iloc[self.positions(MISSING_X), 1] = np.nan
        self.x_const = self.x.copy()
        self.x_const.insert(0, "const", 1.0)

    def positions(self, cells):
        """The positions of the rows that have the (entity, period) cells"""
        wanted = set(cells)
        return [i for i, cell in enumerate(self.cells) if cell in wanted]

    def row_weights(self):
        """The weight of each row, in the order of the rows"""
        return np.array([cell_weight(e, t) for e, t in self.cells])

    def cell_weights(self, index):
        """The weights of the rows of an index that has the labels of this panel"""
        entity = {label: i for i, label in enumerate(self.entities)}
        period = {label: i for i, label in enumerate(self.periods)}
        return np.array([cell_weight(entity[a], period[b]) for a, b in index])

    def build(self, factory, weights):
        """The model, which warns about the observations with missing values"""
        if self.missing == "none":
            return factory(self, weights)
        with pytest.warns(MissingValueWarning):
            return factory(self, weights)


def main_panel(layout, order, missing="none", style="int"):
    return Panel(main_cells(layout, order), missing, style)


MODELS = {
    "PooledOLS": lambda p, w: PooledOLS(p.y, p.x_const, weights=w),
    "PanelOLS-entity": lambda p, w: PanelOLS(p.y, p.x, weights=w, entity_effects=True),
    "PanelOLS-time": lambda p, w: PanelOLS(p.y, p.x, weights=w, time_effects=True),
    "PanelOLS-both": lambda p, w: PanelOLS(
        p.y, p.x, weights=w, entity_effects=True, time_effects=True
    ),
    "RandomEffects": lambda p, w: RandomEffects(p.y, p.x_const, weights=w),
    "BetweenOLS": lambda p, w: BetweenOLS(p.y, p.x_const, weights=w),
    "FirstDifferenceOLS": lambda p, w: FirstDifferenceOLS(p.y, p.x, weights=w),
    "FamaMacBeth": lambda p, w: FamaMacBeth(p.y, p.x_const, weights=w),
}
# The estimates of random effects are not computed by hand
HAND_MODELS = [name for name in MODELS if name != "RandomEffects"]


def wls(y, x, weights):
    """Weighted least squares, which does not use the code of the package"""
    root_w = np.sqrt(weights)
    return np.linalg.lstsq(root_w[:, None] * x, root_w * y, rcond=None)[0]


def dummies(labels):
    """The dummy variables of the distinct values of a vector of labels"""
    labels = np.asarray(labels)
    distinct = pd.unique(labels)
    return (labels[:, None] == distinct[None, :]).astype(float)


def hand_first_difference(index, y, x, weights):
    """Weighted least squares with the differences of adjacent periods of an entity"""
    entity = np.asarray(index.get_level_values(0))
    period = np.asarray(index.get_level_values(1))
    rank = {label: i for i, label in enumerate(np.sort(pd.unique(period)))}
    row = {(e, rank[t]): k for k, (e, t) in enumerate(zip(entity, period, strict=True))}
    dy, dx, dw = [], [], []
    for (e, i), k in row.items():
        j = row.get((e, i - 1))
        if j is not None:
            dy.append(y[k] - y[j])
            dx.append(x[k] - x[j])
            # The variance of a difference is the sum of the variances of its terms
            dw.append(1 / (1 / weights[k] + 1 / weights[j]))
    return wls(np.array(dy), np.array(dx), np.array(dw))


def hand_between(index, y, x, weights):
    """Weighted least squares with the weighted means of the entities"""
    codes = pd.factorize(index.get_level_values(0))[0]
    total = np.bincount(codes, weights=weights)
    y_mean = np.bincount(codes, weights=weights * y) / total
    x_mean = np.column_stack(
        [
            np.bincount(codes, weights=weights * x[:, j]) / total
            for j in range(x.shape[1])
        ]
    )
    return wls(y_mean, x_mean, total)


def hand_fama_macbeth(index, y, x, weights):
    """The average of the weighted least squares estimates of the periods"""
    codes = pd.factorize(index.get_level_values(1))[0]
    estimates = [
        wls(y[codes == t], x[codes == t], weights[codes == t])
        for t in range(codes.max() + 1)
    ]
    return np.mean(estimates, axis=0)


def hand_estimates(model, mod, weights):
    """
    The estimates of a model from the data that it uses and the weights of the rows

    The weights are those of the rows of the model, in their order, after the rows with
    missing values are dropped.
    """
    index = mod.dependent.index
    y = mod.dependent.values2d[:, 0]
    x = mod.exog.values2d
    if model == "BetweenOLS":
        return hand_between(index, y, x, weights)
    if model == "FirstDifferenceOLS":
        return hand_first_difference(index, y, x, weights)
    if model == "FamaMacBeth":
        return hand_fama_macbeth(index, y, x, weights)
    # Least squares with the dummy variables of the effects. The dummy variables of the
    # periods leave out the first period if the entities have them, to avoid collinearity
    entity, period = index.get_level_values(0), index.get_level_values(1)
    columns = [x]
    if model in ("PanelOLS-entity", "PanelOLS-both"):
        columns.append(dummies(entity))
    if model == "PanelOLS-time":
        columns.append(dummies(period))
    if model == "PanelOLS-both":
        columns.append(dummies(period)[:, 1:])
    return wls(y, np.column_stack(columns), weights)[: x.shape[1]]


def assert_same_index(index, expected):
    same = index.equals(expected)
    assert same


def assert_same_weights(mod, expected):
    assert_same_index(mod.weights.dataframe.index, expected.weights.dataframe.index)
    assert_allclose(mod.weights.values2d, expected.weights.values2d, rtol=1e-12)


def assert_weights_of_cells(panel, mod):
    """The weights are those of the cells of the rows, scaled to have a mean of 1"""
    expected = panel.cell_weights(mod.dependent.index)
    assert_allclose(mod.weights.values2d[:, 0], expected / expected.mean(), rtol=1e-12)


def compare_results(res, expected):
    assert res.nobs == expected.nobs
    assert_allclose(res.params, expected.params, rtol=1e-12)
    assert_allclose(res.cov, expected.cov, rtol=1e-12)
    assert_allclose(res.resids, expected.resids, rtol=1e-12)


@pytest.mark.parametrize("missing", MISSING)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("model", MODELS)
def test_weights_for_each_row_are_those_of_the_series(model, layout, order, missing):
    panel = main_panel(layout, order, missing)
    weights = panel.row_weights()
    mod = panel.build(MODELS[model], weights)
    expected = panel.build(MODELS[model], pd.Series(weights, index=panel.index))
    assert_same_weights(mod, expected)
    compare_results(mod.fit(), expected.fit())


@pytest.mark.parametrize("missing", MISSING)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("model", MODELS)
def test_weights_for_each_row_are_attached_to_their_rows(model, layout, order, missing):
    panel = main_panel(layout, order, missing)
    mod = panel.build(MODELS[model], panel.row_weights())
    assert_weights_of_cells(panel, mod)


@pytest.mark.parametrize("missing", MISSING)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("model", HAND_MODELS)
def test_weights_for_each_row_give_weighted_least_squares(
    model, layout, order, missing
):
    panel = main_panel(layout, order, missing)
    mod = panel.build(MODELS[model], panel.row_weights())
    weights = panel.cell_weights(mod.dependent.index)
    res = mod.fit()
    expected = hand_estimates(model, mod, weights)
    assert_allclose(res.params, expected, rtol=1e-6)


FORMS = {
    "array": lambda w: w,
    "list": lambda w: w.tolist(),
    "tuple": lambda w: tuple(w.tolist()),
    "column": lambda w: w[:, None],
    "row": lambda w: w[None, :],
    # A default index has no labels, so that the values are paired by position
    "series": pd.Series,
}


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("missing", MISSING)
@pytest.mark.parametrize("order", ORDERS)
def test_forms_of_weights_for_each_row(order, missing, form):
    panel = main_panel("unbalanced", order, missing)
    weights = panel.row_weights()
    mod = panel.build(MODELS["PooledOLS"], FORMS[form](weights))
    expected = panel.build(MODELS["PooledOLS"], pd.Series(weights, index=panel.index))
    assert_same_weights(mod, expected)
    assert_weights_of_cells(panel, mod)
    compare_results(mod.fit(), expected.fit())


@pytest.mark.parametrize("model", ["PooledOLS", "PanelOLS-time", "FirstDifferenceOLS"])
@pytest.mark.parametrize("order", ORDERS)
def test_weights_for_each_row_with_dates(model, order):
    panel = main_panel("unbalanced", order, "y", style="date")
    weights = panel.row_weights()
    mod = panel.build(MODELS[model], weights)
    expected = panel.build(MODELS[model], pd.Series(weights, index=panel.index))
    assert_same_weights(mod, expected)
    assert_weights_of_cells(panel, mod)
    compare_results(mod.fit(), expected.fit())


@pytest.mark.parametrize("order", ORDERS)
def test_missing_weights_drop_the_observations(order):
    panel = main_panel("unbalanced", order)
    weights = panel.row_weights()
    dropped = [3, 17, 40]
    weights[dropped] = np.nan
    series = pd.Series(weights, index=panel.index)
    with pytest.warns(MissingValueWarning):
        mod = PooledOLS(panel.y, panel.x_const, weights=weights)
    with pytest.warns(MissingValueWarning):
        expected = PooledOLS(panel.y, panel.x_const, weights=series)
    kept = panel.index.delete(dropped)
    assert_same_index(mod.dependent.index, kept)
    assert_same_weights(mod, expected)
    assert_weights_of_cells(panel, mod)
    compare_results(mod.fit(), expected.fit())


@pytest.mark.parametrize("order", ORDERS)
def test_singletons_are_dropped_with_their_weights(order):
    # The last entity has a single observation, which is dropped by the entity effects
    singleton = {(11, 6), (11, 7)}
    cells = [cell for cell in main_cells("unbalanced", order) if cell not in singleton]
    panel = Panel(cells)
    weights = panel.row_weights()
    options = {"entity_effects": True, "singletons": False}
    series = pd.Series(weights, index=panel.index)
    with pytest.warns(SingletonWarning):
        mod = PanelOLS(panel.y, panel.x, weights=weights, **options)
    with pytest.warns(SingletonWarning):
        expected = PanelOLS(panel.y, panel.x, weights=series, **options)
    nkept, expected_nkept = len(mod.dependent.index), len(cells) - 1
    assert nkept == expected_nkept
    assert_same_index(mod.weights.dataframe.index, mod.dependent.index)
    assert_same_weights(mod, expected)
    # The weights are scaled to have a mean of 1 before the singletons are dropped, so
    # that they are compared up to their scale
    found = mod.weights.values2d[:, 0]
    cell = panel.cell_weights(mod.dependent.index)
    assert_allclose(found / found.mean(), cell / cell.mean(), rtol=1e-12)
    compare_results(mod.fit(), expected.fit())


@pytest.mark.parametrize("order", ORDERS)
def test_weights_other_effects_and_clusters_are_paired_with_the_same_rows(order):
    # The three are 1-d arrays with a value for each row of dependent
    panel = main_panel("unbalanced", order, "y")
    weights = panel.row_weights()
    effects = np.array([(e + t) % 3 for e, t in panel.cells])
    clusters = np.array([(2 * e + 3 * t) % 5 for e, t in panel.cells])

    def factory(p, w):
        return PanelOLS(p.y, p.x, weights=w, entity_effects=True, other_effects=effects)

    mod = panel.build(factory, weights)
    labelled = pd.Series(effects, index=panel.index)

    def factory_series(p, w):
        return PanelOLS(
            p.y, p.x, weights=w, entity_effects=True, other_effects=labelled
        )

    expected = panel.build(factory_series, pd.Series(weights, index=panel.index))
    assert_same_weights(mod, expected)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    labelled_clusters = pd.DataFrame({"cluster": clusters}, index=panel.index)
    ref = expected.fit(cov_type="clustered", clusters=labelled_clusters)
    compare_results(res, ref)


FORMULAS = {
    "PooledOLS": (PooledOLS, "y ~ 1 + x0 + x1"),
    "PanelOLS": (PanelOLS, "y ~ x0 + x1 + EntityEffects"),
    "RandomEffects": (RandomEffects, "y ~ 1 + x0 + x1"),
    "BetweenOLS": (BetweenOLS, "y ~ 1 + x0 + x1"),
    "FirstDifferenceOLS": (FirstDifferenceOLS, "y ~ x0 + x1"),
    "FamaMacBeth": (FamaMacBeth, "y ~ 1 + x0 + x1"),
}


@pytest.mark.parametrize("missing", MISSING)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("model", FORMULAS)
def test_from_formula_with_weights_for_each_row(model, order, missing):
    panel = main_panel("unbalanced", order, missing)
    data = pd.concat([panel.y, panel.x], axis=1)
    weights = panel.row_weights()
    cls, formula = FORMULAS[model]

    def factory(p, w):
        return cls.from_formula(formula, data, weights=w)

    mod = panel.build(factory, weights)
    expected = panel.build(factory, pd.Series(weights, index=panel.index))
    assert_same_weights(mod, expected)
    assert_weights_of_cells(panel, mod)
    compare_results(mod.fit(), expected.fit())


EFFECTS = {
    "entity": {"entity_effects": True},
    "time": {"time_effects": True},
    "both": {"entity_effects": True, "time_effects": True},
}


@pytest.mark.parametrize("effects", EFFECTS)
@pytest.mark.parametrize("order", ORDERS)
def test_model_without_regressors_with_weights_for_each_row(order, effects):
    # The number of time periods and entities is that of an exog without columns
    panel = main_panel("unbalanced", order, "y")
    weights = panel.row_weights()

    def factory(p, w):
        exog = pd.DataFrame(index=p.index)
        return PanelOLS(p.y, exog, weights=w, **EFFECTS[effects])

    mod = panel.build(factory, weights)
    expected = panel.build(factory, pd.Series(weights, index=panel.index))
    assert_same_weights(mod, expected)
    assert_weights_of_cells(panel, mod)
    compare_results(mod.fit(), expected.fit())


def values_of_reading(panel, reading):
    """The values that are given, and the weight of each row that they are read as"""
    period_values = 1 + np.arange(NPERIOD) / 2
    entity_values = 1 + np.arange(NENTITY) / 3
    if reading == "rows":
        return panel.row_weights(), panel.row_weights()
    if reading == "periods":
        return period_values, np.array([period_values[t] for _, t in panel.cells])
    return entity_values, np.array([entity_values[e] for e, _ in panel.cells])


# A DataFrame with one column and a default index has no labels, and is read as the
# array with a column is. The name of its column is the name of a variable
COLUMNS = {
    "array": lambda values: values,
    "column": lambda values: values[:, None],
    "frame": lambda values: pd.DataFrame({"weight": values}),
    "frame-default-column": lambda values: pd.DataFrame(values[:, None]),
}


@pytest.mark.parametrize("form", COLUMNS)
@pytest.mark.parametrize("reading", ["rows", "periods", "entities"])
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_number_of_values_selects_the_reading(layout, order, reading, form):
    panel = main_panel(layout, order)
    given, expected = values_of_reading(panel, reading)
    mod = PooledOLS(panel.y, panel.x_const, weights=COLUMNS[form](given))
    assert_weights_equal(mod, expected)


@pytest.mark.parametrize("extra", [-2, -1, 1, 2])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_another_number_of_values_raises_and_names_the_accepted_numbers(layout, extra):
    panel = main_panel(layout, "entity-major")
    nrows = len(panel.cells)
    with pytest.raises(ValueError, match=SHAPE_MESSAGE) as caught:
        PooledOLS(panel.y, panel.x_const, weights=np.ones(nrows + extra))
    message = str(caught.value)
    for accepted in (nrows, NPERIOD, NENTITY):
        assert f"{accepted} values" in message


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_values_raises(layout):
    panel = main_panel(layout, "entity-major")
    with pytest.raises(ValueError, match=SHAPE_MESSAGE):
        PooledOLS(panel.y, panel.x_const, weights=np.ones(0))


def test_a_value_for_each_combination_raises_if_dependent_is_not_balanced():
    # This is not a value for each row. It raised an error from pandas
    panel = main_panel("unbalanced", "entity-major")
    nrows, ncells = len(panel.cells), NENTITY * NPERIOD
    with pytest.raises(ValueError, match=SHAPE_MESSAGE) as caught:
        PooledOLS(panel.y, panel.x_const, weights=np.ones(ncells))
    message = str(caught.value)
    assert "not balanced" in message
    assert f"{nrows} values" in message
    assert f"{nrows} rows and not {ncells}" in message


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", [(NPERIOD, NENTITY - 1), (NPERIOD + 1, NENTITY)])
def test_another_shape_of_2d_weights_raises(layout, shape):
    panel = main_panel(layout, "entity-major")
    with pytest.raises(ValueError, match=SHAPE_MESSAGE) as caught:
        PooledOLS(panel.y, panel.x_const, weights=np.ones(shape))
    message = str(caught.value)
    assert f"{NPERIOD} rows (time periods)" in message


@pytest.mark.parametrize("layout", LAYOUTS)
def test_data_frame_with_one_column_of_the_wrong_length_raises(layout):
    # A column of values is read as the array with a column is, so that its number of
    # values has to be one of the accepted numbers
    panel = main_panel(layout, "entity-major")
    weights = pd.DataFrame({"weight": np.ones(len(panel.cells) + 1)})
    with pytest.raises(ValueError, match=SHAPE_MESSAGE):
        PooledOLS(panel.y, panel.x_const, weights=weights)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_data_frame_with_columns_is_not_a_column_of_values(layout):
    # Two columns are not weights, and the error is the one of the array with two columns
    panel = main_panel(layout, "entity-major")
    values = np.ones((len(panel.cells), 2))
    with pytest.raises(ValueError, match=SHAPE_MESSAGE) as caught:
        PooledOLS(panel.y, panel.x_const, weights=pd.DataFrame(values))
    with pytest.raises(ValueError, match=SHAPE_MESSAGE) as expected:
        PooledOLS(panel.y, panel.x_const, weights=values)
    found, message = str(caught.value), str(expected.value)
    assert found == message


def square_cells(layout):
    """6 entities and 6 periods, so that the numbers of periods and entities are equal"""
    cells = [(e, t) for e in range(6) for t in range(6)]
    if layout == "unbalanced":
        cells = [cell for cell in cells if cell not in {(0, 0), (2, 3), (5, 5)}]
    return cells


@pytest.mark.parametrize("layout", LAYOUTS)
def test_equal_numbers_of_periods_and_entities_are_ambiguous(layout):
    # The number of values does not decide between the periods and the entities. It
    # does if it is the number of rows
    panel = Panel(square_cells(layout))
    with pytest.raises(AmbiguityError, match=AMBIGUITY_MESSAGE):
        PooledOLS(panel.y, panel.x_const, weights=np.ones(6))
    mod = PooledOLS(panel.y, panel.x_const, weights=panel.row_weights())
    assert_weights_of_cells(panel, mod)


# Panels where the number of rows is the number of periods, the number of entities or
# both. The rows are in blocks of an entity or of a period, in an order that is not the
# order of the labels, so that values for each period or entity in the order of the
# labels are not values for each row
ROWS_ARE_PERIODS = [(1, 3), (1, 4), (1, 5), (0, 0), (0, 1), (0, 2)]
ROWS_ARE_ENTITIES = [(2, 2), (5, 2), (0, 0), (3, 0), (1, 1), (4, 1)]
ROWS_ARE_BOTH = [(2, 2), (0, 0), (3, 3), (1, 1)]


@pytest.mark.parametrize(
    "cells", [ROWS_ARE_PERIODS, ROWS_ARE_ENTITIES, ROWS_ARE_BOTH], ids=["T", "N", "T=N"]
)
def test_values_for_each_row_are_read_first_if_dependent_is_not_balanced(cells):
    panel = Panel(cells)
    mod = PooledOLS(panel.y, panel.x_const, weights=panel.row_weights())
    assert_weights_of_cells(panel, mod)


def corner_values(cells):
    """The values 1, 2, ..., one for each row, in the order of the rows"""
    return 1.0 + np.arange(len(cells))


def assert_weights_equal(mod, expected):
    """The weights of the model are the expected weights, scaled to have a mean of 1"""
    expected = np.asarray(expected, dtype=float)
    assert_allclose(mod.weights.values2d[:, 0], expected / expected.mean(), rtol=1e-12)


# An input with labels is read by its labels, and an input without labels, which an
# array, a list, a tuple and a pandas object with a default index are, by the number of
# its values. So a Series indexed by the time periods is a value for each time period,
# and a Series indexed by the entities is a value for each entity, also if dependent is
# not balanced and has as many rows as time periods or entities. The same values without
# labels are a value for each row.
@pytest.mark.parametrize(
    ("cells", "kind"),
    [(ROWS_ARE_PERIODS, "period"), (ROWS_ARE_ENTITIES, "entity")],
    ids=["T", "N"],
)
def test_labels_decide_if_dependent_has_as_many_rows_as_periods_or_entities(
    cells, kind
):
    panel = Panel(cells)
    values = corner_values(cells)
    if kind == "period":
        labels, positions = panel.periods, [t for _, t in cells]
    else:
        labels, positions = panel.entities, [e for e, _ in cells]
    nlabels, nrows = len(labels), len(cells)
    assert nlabels == nrows
    mod = PooledOLS(panel.y, panel.x_const, weights=pd.Series(values, index=labels))
    assert_weights_equal(mod, values[np.asarray(positions)])


@pytest.mark.parametrize(
    "cells", [ROWS_ARE_PERIODS, ROWS_ARE_ENTITIES, ROWS_ARE_BOTH], ids=["T", "N", "T=N"]
)
def test_values_without_labels_are_a_value_for_each_row_in_these_panels(cells):
    panel = Panel(cells)
    values = corner_values(cells)
    frames = (pd.DataFrame({"weight": values}), pd.DataFrame(values[:, None]))
    for unlabelled in (
        values,
        values.tolist(),
        tuple(values),
        pd.Series(values),
        *frames,
    ):
        mod = PooledOLS(panel.y, panel.x_const, weights=unlabelled)
        assert_weights_equal(mod, values)
    # The index of dependent has labels that identify the rows
    series = pd.Series(values, index=panel.index)
    assert_weights_equal(PooledOLS(panel.y, panel.x_const, weights=series), values)


def test_labels_do_not_resolve_equal_numbers_of_periods_and_entities():
    panel = Panel(ROWS_ARE_BOTH)
    values = corner_values(ROWS_ARE_BOTH)
    for labels in (panel.periods, panel.entities):
        series = pd.Series(values, index=labels)
        with pytest.raises(AmbiguityError, match=AMBIGUITY_MESSAGE):
            PooledOLS(panel.y, panel.x_const, weights=series)


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        (ROWS_ARE_PERIODS, "time index of weights"),
        (ROWS_ARE_ENTITIES, "entity index of weights"),
    ],
    ids=["T", "N"],
)
def test_labels_that_are_not_those_of_the_periods_or_entities_raise(cells, message):
    panel = Panel(cells)
    values = corner_values(cells)
    wrong = pd.Series(values, index=[f"label{i}" for i in range(len(cells))])
    with pytest.raises(ValueError, match=message):
        PooledOLS(panel.y, panel.x_const, weights=wrong)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_labels_that_are_not_those_of_the_rows_raise(layout):
    panel = main_panel(layout, "entity-major")
    labels = [f"row{i}" for i in range(len(panel.cells))]
    wrong = pd.Series(panel.row_weights(), index=labels)
    with pytest.raises(ValueError, match="row index of weights"):
        PooledOLS(panel.y, panel.x_const, weights=wrong)


def test_labels_of_the_periods_of_a_single_entity_are_read_as_before():
    # Balanced, and the number of rows is the number of periods
    cells = [(0, t) for t in range(5)]
    panel = Panel(cells)
    values = corner_values(cells)
    series = pd.Series(values, index=panel.periods)
    assert_weights_equal(PooledOLS(panel.y, panel.x_const, weights=series), values)
    wrong = pd.Series(values, index=[f"label{i}" for i in range(5)])
    with pytest.raises(ValueError, match="time index of weights"):
        PooledOLS(panel.y, panel.x_const, weights=wrong)


@pytest.mark.parametrize("reading", ["rows", "periods", "entities"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_data_frame_with_one_column_is_read_as_the_series_with_its_index(
    layout, reading
):
    panel = main_panel(layout, "entity-major")
    given, expected = values_of_reading(panel, reading)
    labels = {"rows": panel.index, "periods": panel.periods, "entities": panel.entities}
    index = labels[reading]
    for name in ("weight", 0):
        weights = pd.DataFrame({name: given}, index=index)
        mod = PooledOLS(panel.y, panel.x_const, weights=weights)
        assert_weights_equal(mod, expected)
