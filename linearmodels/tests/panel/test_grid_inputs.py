"""
Clusters, other effects and weights that are given for each time period and entity

A time by entity grid has a value for each combination of a time period and an entity: a
2-d array or list (time by entity), a 3-d array or list (variable by time by entity), a
DataFrame with the time periods as its index and the entities as its columns, or a
DataArray. The grid is placed on the rows of dependent by time period and entity.

* Clusters and other effects in a grid raised "IndexError: Boolean index has wrong
  length" if dependent was not balanced, which is the case if an entity or a time period
  has missing observations. Weights in a grid raised "weights must have the same number
  of observations as dependent".
* The clusters and the other effects were paired by position with the rows of dependent,
  which is only right if the rows are sorted by entity and then time. If the rows were
  sorted by time and then entity an array was attached to the wrong observations without
  an error, and a DataFrame with labels was rejected.

The reference of the tests is the same model with the values given for each row, as a
Series or DataFrame with the index of dependent, whose results must be identical. The
rows that each value is attached to are also compared with the values that were used to
make the grid, and the estimates with a closed form that does not use the models.
"""

from linearmodels.compat.pandas import ANNUAL_FREQ

import functools

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import xarray as xr

from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.panel.model import _grid_axes
from linearmodels.shared.exceptions import MissingValueWarning

NENTITY = 12
NPERIOD = 8
LAYOUTS = ["balanced", "unbalanced"]
# The order of the rows of dependent. The rows of newest-first are sorted by entity and
# then by time, from the latest period, and those of shuffled are in a random order
ORDERS = ["entity-major", "time-major", "newest-first", "shuffled"]
MISSING = ["none", "y", "x"]
# Cells that are observed in the unbalanced panel, which have a missing value of the
# dependent variable or of a regressor in the panels with missing values
MISSING_Y = [(1, 3), (2, 1), (4, 5), (9, 2), (10, 6)]
MISSING_X = [(1, 1), (4, 2), (7, 3), (9, 6)]


def observed_cells(layout):
    """
    The combinations of an entity and a period that the panel has

    The unbalanced panel has these missing observations. The first entity does not have
    the first two periods, so that the periods do not first appear in the order of time
    when the rows are sorted by entity, and the first entity is not among those that
    have the first period, so that the entities do not first appear in the order of
    their labels when the rows are sorted by period.
    """
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


def grid_order(order, cells):
    """
    The entities and the periods in the order of a grid, without the code of the package

    The grid is sorted if the rows are, and it is in the order of appearance if they are
    not. Entities and periods are the positions 0, 1, ... of the full panel.
    """
    entities = sorted({e for e, _ in cells})
    periods = sorted({t for _, t in cells})
    if order == "newest-first":
        periods = periods[::-1]
    elif order == "shuffled":
        entities = list(dict.fromkeys(e for e, _ in cells))
        periods = list(dict.fromkeys(t for _, t in cells))
    return entities, periods


class Panel:
    """
    A panel of NENTITY entities and NPERIOD periods with missing observations

    Attributes
    ----------
    cells : list of tuple of int
        The (entity, period) of each row, in the order of the rows.
    truth : dict of str to ndarray
        The values of the clusters, effects and weights of every combination of an
        entity and a period, including those that the panel does not have.
    """

    def __init__(self, layout, order, missing="none", style="int"):
        self.layout, self.order, self.missing = layout, order, missing
        rng = np.random.default_rng(20261013)
        # Zero padded, so that the order of the labels is that of the positions
        self.entities = [f"firm{i:02d}" for i in range(NENTITY)]
        if style == "int":
            self.periods = list(range(2001, 2001 + NPERIOD))
        else:
            dates = pd.date_range("2001-12-31", periods=NPERIOD, freq=ANNUAL_FREQ)
            self.periods = list(dates)
        observed = observed_cells(layout)
        cells = [
            (e, t) for e in range(NENTITY) for t in range(NPERIOD) if observed[e, t]
        ]
        if order == "time-major":
            cells.sort(key=lambda cell: (cell[1], cell[0]))
        elif order == "newest-first":
            cells.sort(key=lambda cell: (cell[0], -cell[1]))
        elif order == "shuffled":
            permutation = np.random.default_rng(5).permutation(len(cells))
            cells = [cells[i] for i in permutation]
        self.cells = cells
        self.index = pd.MultiIndex.from_tuples(
            [(self.entities[e], self.periods[t]) for e, t in cells],
            names=["entity", "time"],
        )
        self.grid_entities, self.grid_periods = grid_order(order, cells)

        x = rng.standard_normal((NENTITY, NPERIOD, 2))
        effect = rng.standard_normal(NENTITY)
        shock = rng.standard_normal((NENTITY, NPERIOD))
        y = x @ [1.0, -0.5] + effect[:, None] + shock
        rows = tuple(np.array(cells).T)
        self.x = pd.DataFrame(x[rows], index=self.index, columns=["x0", "x1"])
        self.y = pd.Series(y[rows], index=self.index, name="y")
        if missing == "y":
            self.y.iloc[self.positions(MISSING_Y)] = np.nan
        elif missing == "x":
            self.x.iloc[self.positions(MISSING_X), 1] = np.nan
        self.x_const = self.x.copy()
        self.x_const.insert(0, "const", 1.0)

        e, t = np.meshgrid(np.arange(NENTITY), np.arange(NPERIOD), indexing="ij")
        self.truth = {
            # Clusters that vary over time and over the entities
            "c1": (2 * e + 3 * t) % 5,
            "c2": (e + t) % 4,
            # Clusters of the entities, which BetweenOLS and FirstDifferenceOLS need
            "e1": e % 3,
            "e2": e % 2,
            # Effects that are not nested in the entities, in the periods or in each
            # other
            "o1": (e + t) % 3,
            "o2": (2 * e + 3 * t) % 4,
            "w": 0.5 + ((3 * e + 5 * t) % 7) / 3,
            # Weights of the periods and of the entities
            "wt": (1 + t / 2).astype(float),
            "we": (1 + e / 3).astype(float),
        }

    def positions(self, cells):
        """The positions of the rows that have the (entity, period) cells"""
        wanted = set(cells)
        return [i for i, cell in enumerate(self.cells) if cell in wanted]

    def ids(self, index):
        """The (entity, period) positions of the rows of an index"""
        e = {label: i for i, label in enumerate(self.entities)}
        t = {label: i for i, label in enumerate(self.periods)}
        return [(e[a], t[b]) for a, b in index]

    def row_values(self, name, index=None):
        """The values that the rows have, in the order of the rows"""
        cells = self.cells if index is None else self.ids(index)
        return np.array([self.truth[name][e, t] for e, t in cells])

    def flat(self, *names):
        """The values for each row, as a frame that has the index of dependent"""
        data = {name: self.row_values(name) for name in names}
        return pd.DataFrame(data, index=self.index)

    def grid_labels(self):
        entities = [self.entities[e] for e in self.grid_entities]
        periods = [self.periods[t] for t in self.grid_periods]
        return periods, entities

    def grid_array(self, names):
        """
        The values in the order of the grid, with NaN for the cells without a row

        The array is variable by period by entity. The values that the panel does
        not have would be wrong if they were used, so that they are NaN.
        """
        observed = np.zeros((NENTITY, NPERIOD), dtype=bool)
        for e, t in self.cells:
            observed[e, t] = True
        out = []
        for name in names:
            full = np.where(observed, self.truth[name], np.nan)
            out.append(full[np.ix_(self.grid_entities, self.grid_periods)].T)
        return np.array(out, dtype=float)

    def grid(self, names, form):
        """The values of the variables in a form of a grid"""
        values = self.grid_array(names)
        periods, entities = self.grid_labels()
        coords2d = {"time": periods, "entities": entities}
        coords3d = {"vars": list(names), "time": periods, "entities": entities}
        forms = {
            "frame": lambda: pd.DataFrame(values[0], index=periods, columns=entities),
            "array": lambda: values[0],
            "list": values[0].tolist,
            "array3d": lambda: values,
            "list3d": values.tolist,
            "xarray2d": lambda: xr.DataArray(
                values[0], coords=coords2d, dims=["time", "entities"]
            ),
            "xarray3d": lambda: xr.DataArray(
                values, coords=coords3d, dims=["vars", "time", "entities"]
            ),
        }
        return forms[form]()

    def vector(self, axis, form):
        """The values of the periods or of the entities, in the order of the grid"""
        periods, entities = self.grid_labels()
        if axis == "time":
            values = self.truth["wt"][0, self.grid_periods]
            labels = periods
        else:
            values = self.truth["we"][self.grid_entities, 0]
            labels = entities
        return pd.Series(values, index=labels) if form == "series" else values

    def build(self, factory, **kwargs):
        """The model, which warns about the observations with missing values"""
        if self.missing == "none":
            return factory(self, **kwargs)
        with pytest.warns(MissingValueWarning):
            return factory(self, **kwargs)


# The models that use clusters, with the clusters that they accept: BetweenOLS needs
# clusters that do not vary within an entity, and so does FirstDifferenceOLS, whose
# clusters have to be identical in the periods that are differenced
MODELS = {
    "PooledOLS": (lambda p: PooledOLS(p.y, p.x_const), ("c1", "c2")),
    "PanelOLS-entity": (
        lambda p: PanelOLS(p.y, p.x, entity_effects=True),
        ("c1", "c2"),
    ),
    "PanelOLS-time": (lambda p: PanelOLS(p.y, p.x, time_effects=True), ("c1", "c2")),
    "PanelOLS-both": (
        lambda p: PanelOLS(p.y, p.x, entity_effects=True, time_effects=True),
        ("c1", "c2"),
    ),
    "RandomEffects": (lambda p: RandomEffects(p.y, p.x_const), ("c1", "c2")),
    "BetweenOLS": (lambda p: BetweenOLS(p.y, p.x_const), ("e1", "e2")),
    "FirstDifferenceOLS": (lambda p: FirstDifferenceOLS(p.y, p.x), ("e1", "e2")),
}
ONE_WAY_FORMS = ["frame", "array", "list", "xarray2d", "xarray3d"]
TWO_WAY_FORMS = ["array3d", "list3d", "xarray3d"]


def assert_same_index(index, expected):
    same = index.equals(expected)
    assert same


def compare_results(res, expected):
    # The covariance of two-way clusters is not positive in some of the small samples,
    # so the standard errors, which would warn, are not used
    assert res.nobs == expected.nobs
    assert_allclose(res.params, expected.params, rtol=1e-10)
    assert_allclose(res.cov, expected.cov, rtol=1e-10)
    # The relative rounding error of a residual that is close to 0 is larger, up to
    # 2e-11 if the data change in the last bit, than that of the parameters and the
    # covariance, which is 1e-13
    assert_allclose(res.resids, expected.resids, rtol=1e-10, atol=1e-12)


@functools.lru_cache(maxsize=None)
def reference(layout, order, missing, model, ways):
    """The panel, the model and its results with the clusters for each row"""
    panel = Panel(layout, order, missing)
    factory, names = MODELS[model]
    names = names[:ways]
    mod = panel.build(factory)
    expected = mod.fit(cov_type="clustered", clusters=panel.flat(*names))
    return panel, mod, expected, names


def check_clusters(panel, mod, expected, names, form):
    """The grid is placed on the rows of the model, and gives its results"""
    grid = panel.grid(names, form)
    # Each row has the cluster of its entity and period
    clusters = mod.reformat_clusters(grid)
    assert_same_index(clusters.dataframe.index, mod.dependent.index)
    for k, name in enumerate(names):
        values = panel.row_values(name, mod.dependent.index)
        assert_allclose(clusters.values2d[:, k], values)
    res = mod.fit(cov_type="clustered", clusters=grid)
    compare_results(res, expected)


@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize("ways", [1, 2], ids=["one-way", "two-way"])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_clusters_in_a_grid_are_those_of_the_rows(layout, order, model, ways, missing):
    panel, mod, expected, names = reference(layout, order, missing, model, ways)
    # The other forms of the grid are used with PooledOLS below
    if ways == 1:
        form = "frame" if order == "entity-major" else "xarray3d"
    else:
        form = "array3d" if order == "entity-major" else "xarray3d"
    check_clusters(panel, mod, expected, names, form)


@pytest.mark.parametrize("model", ["PooledOLS", "PanelOLS-both", "BetweenOLS"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_clusters_in_a_grid_with_a_missing_regressor(layout, model):
    panel, mod, expected, names = reference(layout, "time-major", "x", model, 1)
    check_clusters(panel, mod, expected, names, "frame")


ONE_WAY = [(1, form) for form in ONE_WAY_FORMS]
TWO_WAY = [(2, form) for form in TWO_WAY_FORMS]


@pytest.mark.parametrize(("ways", "form"), ONE_WAY + TWO_WAY)
@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_clusters_in_every_form_and_order_of_the_rows(
    layout, order, missing, ways, form
):
    panel, mod, expected, names = reference(layout, order, missing, "PooledOLS", ways)
    check_clusters(panel, mod, expected, names, form)


def manual_clustered_cov(x, resid, groups):
    """(X'X)^-1 S (X'X)^-1 with the sums of the scores of the clusters, one or two-way"""
    scores = x * resid[:, None]

    def meat(codes):
        out = np.zeros((x.shape[1], x.shape[1]))
        for code in np.unique(codes):
            total = scores[codes == code].sum(0)
            out += np.outer(total, total)
        return out

    if len(groups) == 1:
        s = meat(groups[0])
    else:
        first, second = groups
        s = meat(first) + meat(second) - meat(first * (second.max() + 1) + second)
    xpxi = np.linalg.inv(x.T @ x)
    return xpxi @ s @ xpxi


@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize(("ways", "form"), [(1, "frame"), (2, "xarray3d")])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_clustered_covariance_of_a_grid_is_the_manual_covariance(
    layout, order, ways, form, missing
):
    panel = Panel(layout, order, missing)
    mod = panel.build(MODELS["PooledOLS"][0])
    x = mod.exog.values2d
    y = mod.dependent.values2d[:, 0]
    resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    names = ("c1", "c2")[:ways]
    groups = [panel.row_values(name, mod.dependent.index).astype(int) for name in names]
    res = mod.fit(
        cov_type="clustered", clusters=panel.grid(names, form), debiased=False
    )
    assert_allclose(res.cov, manual_clustered_cov(x, resid, groups), rtol=1e-8)


@pytest.mark.parametrize("missing", MISSING)
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_entity_and_time_clusters_with_a_grid(layout, order, missing):
    # The clusters of the entities or the periods are added to the clusters that are
    # given, which have the rows of the model
    panel = Panel(layout, order, missing)
    mod = panel.build(MODELS["PanelOLS-entity"][0])
    for options in ({"cluster_entity": True}, {"cluster_time": True}):
        expected = mod.fit(cov_type="clustered", clusters=panel.flat("c1"), **options)
        res = mod.fit(
            cov_type="clustered", clusters=panel.grid(("c1",), "frame"), **options
        )
        compare_results(res, expected)


EFFECTS = {
    "one": {"names": ("o1",)},
    "one-and-entity": {"names": ("o1",), "entity_effects": True},
    "one-and-time": {"names": ("o1",), "time_effects": True},
    "two": {"names": ("o1", "o2")},
}
EFFECT_FORMS = [(1, form) for form in ONE_WAY_FORMS] + [
    (2, form) for form in TWO_WAY_FORMS
]


def other_effects_model(panel, other_effects, **options):
    return PanelOLS(panel.y, panel.x, other_effects=other_effects, **options)


def least_squares_with_dummies(panel, mod, names, **options):
    """The slopes of the regression on the dummy variables of every effect"""
    x = mod.exog.values2d
    y = mod.dependent.values2d[:, 0]
    index = mod.dependent.index
    columns = [x]
    if options.get("entity_effects"):
        labels = np.array([cell[0] for cell in panel.ids(index)])
        columns.append((labels[:, None] == np.unique(labels)[None, :]) * 1.0)
    if options.get("time_effects"):
        labels = np.array([cell[1] for cell in panel.ids(index)])
        columns.append((labels[:, None] == np.unique(labels)[None, :]) * 1.0)
    for name in names:
        labels = panel.row_values(name, index)
        columns.append((labels[:, None] == np.unique(labels)[None, :]) * 1.0)
    # The dummy variables are collinear, which does not change the slopes
    return np.linalg.lstsq(np.column_stack(columns), y, rcond=None)[0][: x.shape[1]]


def check_categories(panel, mod, names):
    """Each row has the category of its entity and period"""
    cats = mod._other_effect_cats.dataframe
    assert_same_index(cats.index, mod.dependent.index)
    for k, name in enumerate(names):
        values = panel.row_values(name, mod.dependent.index)
        codes = np.unique(values, return_inverse=True)[1]
        assert_allclose(cats.to_numpy()[:, k], codes)


@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize("spec", EFFECTS)
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_other_effects_in_a_grid_are_those_of_the_rows(layout, order, spec, missing):
    panel = Panel(layout, order, missing)
    options = dict(EFFECTS[spec])
    names = options.pop("names")
    expected_mod = panel.build(
        other_effects_model, other_effects=panel.flat(*names), **options
    )
    expected = expected_mod.fit(cov_type="clustered", cluster_entity=True)
    slopes = least_squares_with_dummies(panel, expected_mod, names, **options)
    assert_allclose(expected.params, slopes, rtol=1e-8)
    forms = ["frame", "xarray3d"] if len(names) == 1 else ["array3d", "xarray3d"]
    for form in forms:
        grid = panel.grid(names, form)
        mod = panel.build(other_effects_model, other_effects=grid, **options)
        check_categories(panel, mod, names)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        compare_results(res, expected)
        assert_allclose(res.estimated_effects, expected.estimated_effects, atol=1e-12)
        assert_allclose(res.f_pooled.stat, expected.f_pooled.stat)
        assert_allclose(res.params, slopes, rtol=1e-8)


@functools.lru_cache(maxsize=None)
def reference_effects(layout, order, missing, ways):
    """The panel and the model with the other effects for each row"""
    panel = Panel(layout, order, missing)
    names = ("o1", "o2")[:ways]
    mod = panel.build(other_effects_model, other_effects=panel.flat(*names))
    return panel, mod, names


@pytest.mark.parametrize(("ways", "form"), EFFECT_FORMS)
@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_other_effects_in_every_form_and_order_of_the_rows(
    layout, order, missing, ways, form
):
    panel, expected, names = reference_effects(layout, order, missing, ways)
    mod = panel.build(other_effects_model, other_effects=panel.grid(names, form))
    check_categories(panel, mod, names)
    expected_cats = expected._other_effect_cats.dataframe
    assert_allclose(mod._other_effect_cats.dataframe, expected_cats)
    assert_same_index(mod._other_effect_cats.dataframe.index, expected_cats.index)


WEIGHT_FORMS = [
    ("w", "frame"),
    ("w", "array"),
    ("w", "list"),
    ("w", "xarray2d"),
    ("w", "xarray3d"),
    ("wt", "array"),
    ("wt", "series"),
    ("we", "array"),
    ("we", "series"),
]
WEIGHT_IDS = [f"{name}-{form}" for name, form in WEIGHT_FORMS]
WEIGHT_MODELS = {
    "PooledOLS": lambda p, w: PooledOLS(p.y, p.x_const, weights=w),
    "PanelOLS": lambda p, w: PanelOLS(p.y, p.x, weights=w, entity_effects=True),
    "RandomEffects": lambda p, w: RandomEffects(p.y, p.x_const, weights=w),
    "BetweenOLS": lambda p, w: BetweenOLS(p.y, p.x_const, weights=w),
    "FirstDifferenceOLS": lambda p, w: FirstDifferenceOLS(p.y, p.x, weights=w),
    "FamaMacBeth": lambda p, w: FamaMacBeth(p.y, p.x_const, weights=w),
}


def weights_in_form(panel, name, form):
    if name == "w":
        return panel.grid(("w",), form)
    return panel.vector("time" if name == "wt" else "entity", form)


def flat_weights(panel, name):
    return pd.Series(panel.row_values(name), index=panel.index, name="weight")


@functools.lru_cache(maxsize=None)
def reference_weights(layout, order, missing, name, model):
    """The panel and the model with the weights for each row"""
    panel = Panel(layout, order, missing)
    return panel, panel.build(WEIGHT_MODELS[model], w=flat_weights(panel, name))


@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize(("name", "form"), WEIGHT_FORMS, ids=WEIGHT_IDS)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_weights_in_every_form_and_order_of_the_rows(
    layout, order, name, form, missing
):
    panel, expected = reference_weights(layout, order, missing, name, "PooledOLS")
    mod = panel.build(WEIGHT_MODELS["PooledOLS"], w=weights_in_form(panel, name, form))
    assert_same_index(mod.weights.dataframe.index, expected.weights.dataframe.index)
    assert_allclose(mod.weights.values2d, expected.weights.values2d, rtol=1e-12)
    # Weighted least squares, which does not use the model
    index = mod.dependent.index
    x = mod.exog.values2d
    y = mod.dependent.values2d[:, 0]
    root_w = np.sqrt(panel.row_values(name, index))
    wls = np.linalg.lstsq(root_w[:, None] * x, root_w * y, rcond=None)[0]
    assert_allclose(mod.fit().params, wls, rtol=1e-8)


@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize("model", WEIGHT_MODELS)
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_weights_in_a_grid_are_those_of_the_rows_in_every_model(
    layout, order, model, missing
):
    panel, expected = reference_weights(layout, order, missing, "w", model)
    mod = panel.build(WEIGHT_MODELS[model], w=weights_in_form(panel, "w", "frame"))
    assert_same_index(mod.weights.dataframe.index, expected.weights.dataframe.index)
    assert_allclose(mod.weights.values2d, expected.weights.values2d, rtol=1e-12)
    compare_results(mod.fit(), expected.fit())


def make_labelled(panel, name, form="frame"):
    return panel.grid((name,), form)


def corrupt_reverse_rows(grid):
    return grid.iloc[::-1]


def corrupt_reverse_columns(grid):
    return grid[grid.columns[::-1]]


def corrupt_shift_periods(grid):
    out = grid.copy()
    out.index = np.roll(out.index, 1)
    return out


def corrupt_rename_entities(grid):
    out = grid.copy()
    out.columns = [f"{label}x" for label in out.columns]
    return out


CORRUPTIONS = {
    "reverse-rows": corrupt_reverse_rows,
    "reverse-columns": corrupt_reverse_columns,
    "shift-periods": corrupt_shift_periods,
    "rename-entities": corrupt_rename_entities,
}
MISALIGNED = r"The time and entity index of {} does not match dependent\."


@pytest.mark.parametrize("corruption", CORRUPTIONS)
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_grids_with_labels_must_use_the_labels_of_dependent(layout, order, corruption):
    # The label checks of the weights apply to the clusters and the other effects, also
    # if dependent is not balanced
    panel = Panel(layout, order)
    mod = PooledOLS(panel.y, panel.x_const)
    bad = CORRUPTIONS[corruption](make_labelled(panel, "c1"))
    with pytest.raises(ValueError, match=MISALIGNED.format("clusters")):
        mod.reformat_clusters(bad)
    with pytest.raises(ValueError, match=MISALIGNED.format("clusters")):
        mod.fit(cov_type="clustered", clusters=bad)
    bad = CORRUPTIONS[corruption](make_labelled(panel, "o1"))
    with pytest.raises(ValueError, match=MISALIGNED.format("other_effects")):
        PanelOLS(panel.y, panel.x, other_effects=bad)
    bad = CORRUPTIONS[corruption](make_labelled(panel, "w"))
    with pytest.raises(ValueError, match=MISALIGNED.format("weights")):
        PooledOLS(panel.y, panel.x_const, weights=bad)


@pytest.mark.parametrize("dim", ["time", "entities"])
@pytest.mark.parametrize("form", ["xarray2d", "xarray3d"])
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_data_arrays_must_use_the_labels_of_dependent(layout, order, form, dim):
    panel = Panel(layout, order)
    mod = PooledOLS(panel.y, panel.x_const)
    grid = panel.grid(("c1",), form)
    bad = grid.isel({dim: slice(None, None, -1)})
    with pytest.raises(ValueError, match=MISALIGNED.format("clusters")):
        mod.reformat_clusters(bad)
    effects = panel.grid(("o1",), form).isel({dim: slice(None, None, -1)})
    with pytest.raises(ValueError, match=MISALIGNED.format("other_effects")):
        PanelOLS(panel.y, panel.x, other_effects=effects)


@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_grids_without_labels_are_paired_by_position_in_the_grid(layout, order):
    # An array, and a DataFrame that only has default labels, have nothing to compare
    panel = Panel(layout, order)
    mod = PooledOLS(panel.y, panel.x_const)
    expected = mod.fit(cov_type="clustered", clusters=panel.flat("c1"))
    frame = make_labelled(panel, "c1")
    default = pd.DataFrame(frame.to_numpy())
    assert_allclose(
        mod.fit(cov_type="clustered", clusters=default).cov, expected.cov, rtol=1e-12
    )
    # Reversed values in the same positions are not the values of the rows. The cells
    # without a row are filled, since reversing moves their missing values to rows,
    # and the cluster of a row cannot be missing
    reversed_grid = pd.DataFrame(np.nan_to_num(frame.to_numpy())[::-1])
    other = mod.fit(cov_type="clustered", clusters=reversed_grid)
    unchanged = np.allclose(other.cov, expected.cov)
    assert not unchanged


@pytest.mark.parametrize("layout", LAYOUTS)
def test_grid_with_the_wrong_shape_is_rejected(layout):
    panel = Panel(layout, "entity-major")
    mod = PooledOLS(panel.y, panel.x_const)
    grid = make_labelled(panel, "c1")
    for bad in (grid.iloc[:-1], grid.iloc[:, :-1], grid.iloc[:, :-1].to_numpy()):
        with pytest.raises(ValueError, match="same number of entities and time"):
            mod.fit(cov_type="clustered", clusters=bad)
        with pytest.raises(ValueError, match="same number of entities and time"):
            PanelOLS(panel.y, panel.x, other_effects=bad)
    # The same number of cells, with the periods and the entities swapped
    swapped = grid.to_numpy().T
    assert swapped.size == grid.size
    with pytest.raises(ValueError, match="same number of entities and time"):
        mod.fit(cov_type="clustered", clusters=swapped)


def appearance(values):
    return list(dict.fromkeys(values))


@pytest.mark.parametrize("missing", ["none", "y"])
@pytest.mark.parametrize("order", ["entity-major", "time-major", "newest-first"])
def test_unbalanced_grid_is_in_the_order_of_the_rows_not_of_their_appearance(
    order, missing
):
    # The first entity does not have the first periods, so that the periods first
    # appear in an order that is not the order of time if the rows are sorted by
    # entity. The grid is in the order of the periods, and a grid in the order of
    # appearance, which has other periods in its rows, is rejected
    panel = Panel("unbalanced", order, missing)
    mod = panel.build(MODELS["PooledOLS"][0])
    periods = [panel.periods[t] for _, t in panel.cells]
    entities = [panel.entities[e] for e, _ in panel.cells]
    grid = make_labelled(panel, "c1")
    first = appearance(periods), appearance(entities)
    found = list(grid.index), list(grid.columns)
    if order == "newest-first":
        # Newest first is the order of appearance, since the first entity has the
        # last period
        assert found[0] == first[0]
    else:
        assert found != first
        by_appearance = grid.reindex(index=first[0], columns=first[1])
        with pytest.raises(ValueError, match="time and entity index of clusters"):
            mod.fit(cov_type="clustered", clusters=by_appearance)
    expected = mod.fit(cov_type="clustered", clusters=panel.flat("c1"))
    compare_results(mod.fit(cov_type="clustered", clusters=grid), expected)


@pytest.mark.parametrize("seed", range(40))
def test_balanced_panels_use_the_order_in_which_the_rows_appear(seed):
    # The order of a grid is the order of appearance if the panel is balanced, whatever
    # the order of the rows is, as it is for the weights of earlier versions
    rng = np.random.default_rng(seed)
    entities = rng.permutation(np.arange(5, 5 + rng.integers(2, 6)))
    periods = rng.permutation(np.arange(10, 10 + rng.integers(2, 6)))
    cells = [(e, t) for e in entities for t in periods]
    kind = seed % 4
    if kind == 1:
        cells.sort()
    elif kind == 2:
        cells.sort(key=lambda cell: (cell[1], cell[0]))
    elif kind == 3:
        cells = [cells[i] for i in rng.permutation(len(cells))]
    index = pd.MultiIndex.from_tuples(cells)
    found_entities, found_periods = (list(axis) for axis in _grid_axes(index))
    expected_entities = appearance(e for e, _ in cells)
    expected_periods = appearance(t for _, t in cells)
    assert found_entities == expected_entities
    assert found_periods == expected_periods


@pytest.mark.parametrize("seed", range(40))
def test_sorted_unbalanced_panels_use_sorted_axes(seed):
    # The rows are sorted by entity, or by period, or newest first, and observations
    # are missing. The axes are sorted in the direction of the rows
    rng = np.random.default_rng(seed)
    nentity, nperiod = rng.integers(3, 8), rng.integers(3, 8)
    keep = rng.random((nentity, nperiod)) < 0.7
    for entity in range(nentity):
        keep[entity, rng.choice(nperiod, 2, replace=False)] = True
    cells = [(e, t) for e in range(nentity) for t in range(nperiod) if keep[e, t]]
    kind = seed % 3
    if kind == 1:
        cells.sort(key=lambda cell: (cell[1], cell[0]))
    elif kind == 2:
        cells.sort(key=lambda cell: (cell[0], -cell[1]))
    index = pd.MultiIndex.from_tuples(cells)
    entities, periods = (list(axis) for axis in _grid_axes(index))
    present_e = sorted({e for e, _ in cells})
    present_t = sorted({t for _, t in cells})
    expected_periods = present_t[::-1] if kind == 2 else present_t
    assert entities == present_e
    assert periods == expected_periods


def test_axes_of_labels_that_cannot_be_ordered_are_in_order_of_appearance():
    # Entities that are strings and integers have no order. The periods of every
    # entity are in order, so that they are in order of time and not of appearance
    cells = [("a", 3), ("a", 4), (2, 1), (2, 3), ("b", 1)]
    index = pd.MultiIndex.from_tuples(cells)
    entities, periods = (list(axis) for axis in _grid_axes(index))
    assert entities == ["a", 2, "b"]
    assert periods == [1, 3, 4]


@pytest.mark.parametrize("order", ["entity-major", "time-major"])
def test_dates_are_placed_by_time_period_and_entity(order):
    panel = Panel("unbalanced", order, style="date")
    mod = PanelOLS(panel.y, panel.x, entity_effects=True)
    expected = mod.fit(cov_type="clustered", clusters=panel.flat("c1"))
    for form in ("frame", "array", "list"):
        grid = panel.grid(("c1",), form)
        compare_results(mod.fit(cov_type="clustered", clusters=grid), expected)
    effects = PanelOLS(panel.y, panel.x, other_effects=panel.flat("o1"))
    grid_effects = PanelOLS(
        panel.y, panel.x, other_effects=panel.grid(("o1",), "frame")
    )
    assert_allclose(grid_effects.fit().params, effects.fit().params, rtol=1e-12)
    weights = PooledOLS(panel.y, panel.x_const, weights=panel.grid(("w",), "frame"))
    flat = PooledOLS(panel.y, panel.x_const, weights=flat_weights(panel, "w"))
    assert_allclose(weights.fit().params, flat.fit().params, rtol=1e-12)


def test_unlabelled_dependent_with_missing_values():
    # A time by entity array has a row for every period and entity, and the rows with
    # missing values are dropped. The clusters in a grid are placed on all of the rows
    rng = np.random.default_rng(7)
    nperiod, nentity = 6, 9
    x = rng.standard_normal((2, nperiod, nentity))
    y = x[0] - 0.5 * x[1] + rng.standard_normal((nperiod, nentity))
    y[[1, 3, 4], [0, 5, 7]] = np.nan
    x[1, [2, 2], [3, 8]] = np.nan
    t, e = np.meshgrid(np.arange(nperiod), np.arange(nentity), indexing="ij")
    clusters = (2 * e + 3 * t) % 5
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(y, x, entity_effects=True)
    # The rows are stacked by entity
    flat = clusters.T.ravel()
    expected = mod.fit(cov_type="clustered", clusters=flat)
    for grid in (clusters, clusters.tolist(), pd.DataFrame(clusters)):
        res = mod.fit(cov_type="clustered", clusters=grid)
        compare_results(res, expected)
    kept = mod.reformat_clusters(clusters).values2d[:, 0]
    assert_allclose(kept, flat[mod.not_null])


@pytest.mark.parametrize("kind", ["text", "category"])
@pytest.mark.parametrize("order", ["entity-major", "time-major"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_text_and_categories_in_a_grid_identify_the_groups(layout, order, kind):
    # They are not converted to dummy variables when they are placed on the rows
    panel = Panel(layout, order)
    letters = np.array(list("abcde"))
    flat = pd.DataFrame(
        {"c1": letters[panel.row_values("c1").astype(int)]}, index=panel.index
    )
    grid = panel.grid(("c1",), "frame")
    values = grid.to_numpy()
    text = np.where(np.isnan(values), None, letters[np.nan_to_num(values).astype(int)])
    text_grid = pd.DataFrame(text, index=grid.index, columns=grid.columns)
    if kind == "category":
        text_grid = text_grid.astype("category")
    mod = PooledOLS(panel.y, panel.x_const)
    expected = mod.fit(cov_type="clustered", clusters=flat)
    compare_results(mod.fit(cov_type="clustered", clusters=text_grid), expected)
    effects = PanelOLS(panel.y, panel.x, other_effects=flat).fit()
    grid_effects = PanelOLS(panel.y, panel.x, other_effects=text_grid).fit()
    assert_allclose(grid_effects.params, effects.params, rtol=1e-12)
