"""
Dates are compared by value when the labels of the inputs are checked.

pandas 2 keeps dates at the resolution of the data, e.g., datetime64[us] for
dates from a Parquet file or from a NumPy array of datetime64[us]. In pandas 2
``Index.equals`` treats the same dates at different resolutions as different
labels, and in pandas 3 it does not. The estimators check that the inputs that
carry labels use the labels of the dependent variable. Inputs that have the
same dates at different resolutions must be accepted in both versions, and must
give the results of inputs that all have the same resolution. Different dates,
and the same dates in a different order, are still rejected.

The dates are built from NumPy arrays. pandas before 2.0 only has nanoseconds
and converts these arrays to nanoseconds, so the tests also hold there.
"""

from contextlib import nullcontext
from functools import cache
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels import OLS
from linearmodels.asset_pricing import (
    LinearFactorModel,
    LinearFactorModelGMM,
    TradedFactorModel,
)
from linearmodels.iv import (
    IV2SLS,
    IVGMM,
    IVGMMCUE,
    IVJIVE,
    IVLIML,
    AbsorbingLS,
    Interaction,
)
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared import alignment
from linearmodels.shared.alignment import check_row_alignment, differs
from linearmodels.system import IV3SLS, SUR, IVSystemGMM

UNITS = ["s", "ms", "us", "ns"]
# (resolution of the first input, resolution of the second input)
UNIT_PAIRS = [("us", "ns"), ("ns", "us"), ("s", "ms")]
# Resolutions other than nanoseconds only exist in pandas 2.0 and later
HAS_UNITS = hasattr(pd.DatetimeIndex, "as_unit")
needs_units = pytest.mark.skipif(not HAS_UNITS, reason="pandas before 2.0 is ns only")


@cache
def xarray_converts_to_nanoseconds():
    """Whether xarray warns when it converts the dates of a DataArray to nanoseconds

    xarray 2022.11.0 to 2025.1.1 convert coordinates that are datetime64[s], [ms] or
    [us] to datetime64[ns] and warn. The other versions keep the unit. The behavior is
    probed so that the tests expect the warning exactly where xarray issues it.
    """
    xr = pytest.importorskip("xarray")
    coords = {"time": np.array(["2000-01-01"], dtype="datetime64[s]")}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        xr.DataArray([1.0], dims=["time"], coords=coords)
    return any("non-nanosecond" in str(w.message) for w in record)


def data_array_warns(unit):
    """Context for the construction of a DataArray with dates that have the unit"""
    if unit != "ns" and xarray_converts_to_nanoseconds():
        return pytest.warns(UserWarning, match="Converting non-nanosecond precision")
    return nullcontext()


def dates(unit, n, shift=0, tz=None):
    """Consecutive days from a NumPy array with the given resolution"""
    first = np.datetime64("2000-01-01") + np.timedelta64(shift, "D")
    days = first + np.arange(n).astype("timedelta64[D]")
    index = pd.DatetimeIndex(days.astype(f"datetime64[{unit}]"))
    return index if tz is None else index.tz_localize(tz)


def deltas(unit, n, shift=0):
    """Consecutive hours as time deltas from a NumPy array with the resolution"""
    hours = (shift + np.arange(n)).astype("timedelta64[h]")
    return pd.TimedeltaIndex(hours.astype(f"timedelta64[{unit}]"))


def entity_time(unit, entities=("a", "b"), n=4, shift=0, tz=None):
    return pd.MultiIndex.from_product([list(entities), dates(unit, n, shift, tz)])


def assert_same(left, right):
    """The labels are the same, whichever of them is the reference"""
    forward = differs(left, right)
    backward = differs(right, left)
    assert not forward
    assert not backward


def assert_different(left, right):
    """The labels are different, whichever of them is the reference"""
    forward = differs(left, right)
    backward = differs(right, left)
    assert forward
    assert backward


def assert_same_results(result, expected, resids=True, rtol=1e-10):
    assert_allclose(result.params, expected.params, rtol=rtol)
    assert_allclose(result.std_errors, expected.std_errors, rtol=rtol)
    if resids:
        assert_allclose(result.resids, expected.resids, rtol=rtol, atol=1e-12)


# The comparison of two sets of labels


@pytest.mark.parametrize("left", UNITS)
@pytest.mark.parametrize("right", UNITS)
def test_same_dates_at_any_resolutions(left, right):
    assert_same(dates(left, 5), dates(right, 5))
    assert_same(entity_time(left), entity_time(right))


@pytest.mark.parametrize(("left", "right"), UNIT_PAIRS)
def test_different_dates_at_different_resolutions(left, right):
    base = dates(left, 5)
    one_different = dates(right, 2).append(dates(right, 3, shift=1000))
    assert_different(base, one_different)
    assert_different(base, dates(right, 5, shift=1))
    assert_different(base, dates(right, 5)[::-1])
    assert_different(base, dates(right, 4))
    assert_different(base, dates(right, 6))


@pytest.mark.parametrize(("left", "right"), UNIT_PAIRS)
def test_multiindex_with_dates_in_the_second_level(left, right):
    base = entity_time(left)
    assert_different(base, entity_time(right, shift=1))
    assert_different(base, entity_time(right)[::-1])
    assert_different(base, entity_time(right, entities=("a", "c")))
    assert_different(base, entity_time(right, n=3))
    other = entity_time(right)
    swapped = pd.MultiIndex.from_arrays(
        [other.get_level_values(1), other.get_level_values(0)]
    )
    assert_different(base, swapped)


@pytest.mark.parametrize(("left", "right"), UNIT_PAIRS)
def test_multiindex_with_dates_in_other_levels(left, right):
    def build(unit):
        return pd.MultiIndex.from_product([dates(unit, 3), [1, 2]])

    assert_same(build(left), build(right))
    assert_different(build(left), build(right)[::-1])
    # Dates in both levels, where the resolutions of the levels are different
    both = pd.MultiIndex.from_product([dates(left, 3), dates(right, 2)])
    swapped = pd.MultiIndex.from_product([dates(right, 3), dates(left, 2)])
    assert_same(both, swapped)
    # A third level
    three = pd.MultiIndex.from_product([["x", "y"], [1], dates(left, 3)])
    other = pd.MultiIndex.from_product([["x", "y"], [1], dates(right, 3)])
    assert_same(three, other)
    # A different number of levels
    assert_different(entity_time(left), three)


@pytest.mark.parametrize("tz", ["UTC", "US/Eastern"])
@pytest.mark.parametrize(("left", "right"), UNIT_PAIRS)
def test_time_zones_are_not_compared_by_instant(tz, left, right):
    aware = dates(left, 4, tz=tz)
    assert_same(aware, dates(right, 4, tz=tz))
    assert_same(entity_time(left, tz=tz), entity_time(right, tz=tz))
    assert_different(aware, dates(right, 4, shift=1, tz=tz))
    # A date with and without a time zone are different labels in pandas 3
    assert_different(aware, dates(right, 4))
    assert_different(entity_time(left, tz=tz), entity_time(right))
    # and so are the same instants in another time zone
    elsewhere = dates(right, 4, tz=tz).tz_convert("Asia/Tokyo")
    assert_different(aware, elsewhere)


@pytest.mark.parametrize("left", UNITS)
@pytest.mark.parametrize("right", UNITS)
def test_time_deltas(left, right):
    base = deltas(left, 5)
    assert_same(base, deltas(right, 5))
    assert_same(
        pd.MultiIndex.from_product([["a", "b"], base]),
        pd.MultiIndex.from_product([["a", "b"], deltas(right, 5)]),
    )
    assert_different(base, deltas(right, 5, shift=1))
    assert_different(base, deltas(right, 5)[::-1])
    assert_different(base, deltas(right, 4))
    # A date is not a time delta
    assert_different(dates(left, 5), deltas(right, 5))


def test_the_resolution_is_converted_and_not_the_integers():
    # The same numbers are different values when the resolutions are different
    numbers = np.array([1, 2, 3])
    micro = pd.TimedeltaIndex(numbers.astype("timedelta64[us]"))
    nano = pd.TimedeltaIndex(numbers.astype("timedelta64[ns]"))
    assert_different(micro, nano)
    seconds = pd.DatetimeIndex(numbers.astype("datetime64[s]"))
    nanoseconds = pd.DatetimeIndex(numbers.astype("datetime64[ns]"))
    assert_different(seconds, nanoseconds)


FAR_FUTURE = ["2900-01-01", "3000-01-01", "3100-01-01"]
FAR_PAST = ["1000-01-01", "1100-01-01", "1200-01-01"]


def far(unit, values):
    return pd.DatetimeIndex(np.array(values, dtype=f"datetime64[{unit}]"))


def far_index(unit, year, n, entities=None):
    """Consecutive days from January 1 of the year, which nanoseconds cannot hold"""
    days = np.datetime64(f"{year}-01-01") + np.arange(n).astype("timedelta64[D]")
    times = pd.DatetimeIndex(days.astype(f"datetime64[{unit}]"))
    if entities is None:
        return times
    return pd.MultiIndex.from_product([entities, times], names=["entity", "time"])


@needs_units
@pytest.mark.parametrize("values", [FAR_FUTURE, FAR_PAST], ids=["after", "before"])
def test_dates_outside_the_nanosecond_range(values):
    # They are compared at the finer of the resolutions, which can hold them
    assert_same(far("s", values), far("us", values))
    assert_same(far("s", values), far("ms", values))
    assert_same(far("ms", values), far("us", values))
    different = far("us", [*values[:2], "3300-01-01"])
    assert_different(far("s", values), different)
    assert_different(far("s", values), far("us", values)[::-1])
    # Nanoseconds cannot hold them, so they are none of the dates that it holds
    within = dates("ns", 3)
    assert_different(far("s", values), within)
    assert_different(far("us", values), within)


@needs_units
@pytest.mark.parametrize("values", [FAR_FUTURE, FAR_PAST], ids=["after", "before"])
def test_multiindex_with_dates_outside_the_nanosecond_range(values):
    def build(unit, dates_):
        return pd.MultiIndex.from_product([["a", "b"], far(unit, dates_)])

    assert_same(build("s", values), build("us", values))
    assert_different(build("s", values), build("us", values)[::-1])
    assert_different(build("s", values), build("us", [*values[:2], "3300-01-01"]))
    within = pd.MultiIndex.from_product([["a", "b"], dates("ns", 3)])
    assert_different(build("s", values), within)
    assert_different(build("us", values), within)


@needs_units
def test_unused_values_of_a_level_are_not_compared():
    # A level of the first has a date that nanoseconds cannot hold, but no row
    # uses it, and so the rows have the same labels
    first = pd.MultiIndex(
        levels=[["a"], far("s", ["2000-01-31", "3000-01-01"])], codes=[[0, 0], [0, 0]]
    )
    second = pd.MultiIndex(
        levels=[["a"], far("ns", ["2000-01-31"])], codes=[[0, 0], [0, 0]]
    )
    assert_same(first, second)


def test_missing_dates():
    left = pd.DatetimeIndex(np.array(["2000-01-01", "NaT"], dtype="datetime64[us]"))
    right = pd.DatetimeIndex(np.array(["2000-01-01", "NaT"], dtype="datetime64[ns]"))
    moved = pd.DatetimeIndex(np.array(["NaT", "2000-01-01"], dtype="datetime64[ns]"))
    assert_same(left, right)
    assert_different(left, moved)
    assert_same(
        pd.MultiIndex.from_arrays([["a", "b"], left]),
        pd.MultiIndex.from_arrays([["a", "b"], right]),
    )


def test_other_labels_are_compared_as_before():
    assert_same(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]))
    assert_same(pd.Index(list("abc")), pd.Index(list("abc")))
    assert_different(pd.Index([1, 2, 3]), pd.Index([1, 2, 4]))
    assert_different(pd.Index(list("abc")), pd.Index(list("cba")))
    assert_different(pd.Index(list("abc")), pd.Index(list("abcd")))
    # Dates are not numbers or text
    assert_different(dates("us", 3), pd.Index([10, 11, 12]))
    assert_different(dates("us", 3), pd.Index(["2000-01-01", "2000-01-02", "x"]))
    assert_different(entity_time("us"), pd.Index(list(range(10, 18))))
    same = pd.MultiIndex.from_product([["a", "b"], [1, 2]])
    assert_same(same, pd.MultiIndex.from_product([["a", "b"], [1, 2]]))
    assert_different(same, pd.MultiIndex.from_product([["a", "b"], [1, 3]]))


@pytest.mark.parametrize(("left", "right"), UNIT_PAIRS)
def test_check_row_alignment(left, right):
    match = "row index of exog does not match dependent"
    cases = [
        (dates(left, 5), dates(right, 5)),
        (entity_time(left), entity_time(right)),
        (dates(left, 5, tz="UTC"), dates(right, 5, tz="UTC")),
        (deltas(left, 5), deltas(right, 5)),
    ]
    for expected, labels in cases:
        check_row_alignment("exog", labels, expected)
        with pytest.raises(ValueError, match=match):
            check_row_alignment("exog", labels[::-1], expected)
    with pytest.raises(ValueError, match=match):
        check_row_alignment("exog", dates(right, 5, shift=1), dates(left, 5))
    with pytest.raises(ValueError, match=match):
        check_row_alignment("exog", entity_time(right, shift=1), entity_time(left))


@pytest.mark.parametrize(("left", "right"), UNIT_PAIRS)
def test_same_labels(left, right):
    cases = [
        (dates(left, 4), dates(right, 4)),
        (entity_time(left), entity_time(right)),
        (deltas(left, 4), deltas(right, 4)),
    ]
    for first, second in cases:
        same = alignment.same_labels(first, second)
        reversed_labels = alignment.same_labels(first, second[::-1])
        assert same
        assert not reversed_labels


def test_same_labels_compares_a_default_index():
    # differs does not, since the rows of a default index do not have labels
    default = pd.RangeIndex(3)
    same = alignment.same_labels(default, pd.Index([0, 1, 2]))
    different = alignment.same_labels(default, dates("us", 3))
    not_compared = differs(default, dates("us", 3))
    assert same
    assert not different
    assert not not_compared


# Panel models

NENTITY, NTIME = 6, 8
ENTITIES = [f"firm{i}" for i in range(NENTITY)]
OTHER_ENTITIES = [f"unit{i}" for i in range(NENTITY)]
PANEL_MODELS = [
    pytest.param(PooledOLS, {}, id="PooledOLS"),
    pytest.param(
        PanelOLS, {"entity_effects": True, "time_effects": True}, id="PanelOLS"
    ),
    pytest.param(RandomEffects, {}, id="RandomEffects"),
    pytest.param(BetweenOLS, {}, id="BetweenOLS"),
    pytest.param(FirstDifferenceOLS, {}, id="FirstDifferenceOLS"),
    pytest.param(FamaMacBeth, {}, id="FamaMacBeth"),
]
# Fama-MacBeth has its own covariance estimators and does not use clusters
CLUSTER_MODELS = PANEL_MODELS[:-1]
WEIGHT_SHAPES = ["flat", "grid", "time"]


class PanelInputs:
    """The values of a panel, where the dates of each input have a resolution"""

    def __init__(self):
        rng = np.random.default_rng(20261013)
        nobs = NENTITY * NTIME
        self.x = rng.standard_normal((nobs, 2))
        self.y = self.x @ np.array([0.5, -1.0]) + rng.standard_normal(nobs)
        self.w_flat = rng.chisquare(5, nobs) + 0.5
        self.w_grid = rng.chisquare(5, (NTIME, NENTITY)) + 0.5
        self.w_time = rng.chisquare(5, NTIME) + 0.5
        # Constant within an entity, as every covariance estimator requires
        self.clusters = np.repeat(np.arange(NENTITY) % 3, NTIME)
        self.other = np.arange(nobs) % 3

    @staticmethod
    def index(unit, shift=0, entities=ENTITIES):
        return pd.MultiIndex.from_product(
            [entities, dates(unit, NTIME, shift)], names=["entity", "time"]
        )

    def dependent(self, unit="ns", **kwargs):
        return pd.Series(self.y, index=self.index(unit, **kwargs), name="y")

    def exog(self, unit="ns", **kwargs):
        index = self.index(unit, **kwargs)
        return pd.DataFrame(self.x, index=index, columns=["a", "b"])

    def weights(self, shape, unit="ns", shift=0):
        if shape == "flat":
            return pd.Series(self.w_flat, index=self.index(unit, shift), name="w")
        index = dates(unit, NTIME, shift)
        if shape == "grid":
            return pd.DataFrame(self.w_grid, index=index, columns=ENTITIES)
        return pd.Series(self.w_time, index=index, name="w")

    def expected_weights(self, shape):
        """The weight of each row of the panel, which has all times of an entity"""
        if shape == "flat":
            return self.w_flat
        if shape == "grid":
            return self.w_grid.T.ravel()
        return np.tile(self.w_time, NENTITY)

    def clusters_frame(self, shape, unit="ns", shift=0):
        if shape == "flat":
            return pd.DataFrame({"cl": self.clusters}, index=self.index(unit, shift))
        wide = self.clusters.reshape(NENTITY, NTIME).T
        index = dates(unit, NTIME, shift)
        return pd.DataFrame(wide, index=index, columns=ENTITIES)

    def other_effects(self, unit="ns", shift=0):
        return pd.DataFrame({"c": self.other}, index=self.index(unit, shift))


@pytest.fixture(scope="module")
def panel():
    return PanelInputs()


def fit_panel(model, options, dependent, exog, clusters=None, **kwargs):
    mod = model(dependent, exog, **options, **kwargs)
    if clusters is None:
        return mod.fit()
    return mod.fit(cov_type="clustered", clusters=clusters)


@pytest.mark.parametrize(("model", "options"), PANEL_MODELS)
@pytest.mark.parametrize(("y_unit", "x_unit"), UNIT_PAIRS)
def test_panel_dependent_and_exog(panel, model, options, y_unit, x_unit):
    expected = fit_panel(model, options, panel.dependent(), panel.exog())
    y, x = panel.dependent(y_unit), panel.exog(x_unit)
    assert_same_results(fit_panel(model, options, y, x), expected)


@pytest.mark.parametrize("model", [PooledOLS, RandomEffects])
def test_panel_results_use_the_dates_of_dependent(panel, model):
    y, x = panel.dependent("us"), panel.exog("ns")
    result = model(y, x).fit()
    resids_match = result.resids.index.equals(y.index)
    fitted_match = result.fitted_values.index.equals(y.index)
    model_match = result.model.dependent.index.equals(y.index)
    assert resids_match
    assert fitted_match
    assert model_match


@pytest.mark.parametrize(("model", "options"), PANEL_MODELS)
@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
@pytest.mark.parametrize("unit", ["us", "s"])
def test_panel_weights(panel, model, options, shape, unit):
    y, x = panel.dependent(), panel.exog()
    expected = fit_panel(model, options, y, x, weights=panel.weights(shape))
    result = fit_panel(model, options, y, x, weights=panel.weights(shape, unit))
    assert_same_results(result, expected)


@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
@pytest.mark.parametrize(("y_unit", "w_unit"), UNIT_PAIRS)
def test_panel_weights_match_weighted_least_squares(panel, shape, y_unit, w_unit):
    root_w = np.sqrt(panel.expected_weights(shape))
    wx, wy = root_w[:, None] * panel.x, root_w * panel.y
    expected = np.linalg.lstsq(wx, wy, rcond=None)[0]
    weights = panel.weights(shape, w_unit)
    result = PooledOLS(panel.dependent(y_unit), panel.exog("ms"), weights=weights)
    assert_allclose(result.fit().params, expected)


@pytest.mark.parametrize(("model", "options"), CLUSTER_MODELS)
@pytest.mark.parametrize("shape", ["flat", "grid"])
@pytest.mark.parametrize("unit", ["us", "s"])
def test_panel_clusters(panel, model, options, shape, unit):
    y, x = panel.dependent(), panel.exog()
    clusters = panel.clusters_frame(shape)
    expected = fit_panel(model, options, y, x, clusters=clusters)
    clusters = panel.clusters_frame(shape, unit)
    assert_same_results(fit_panel(model, options, y, x, clusters=clusters), expected)


def test_panel_clusters_and_cluster_entity(panel):
    kwargs = {"cov_type": "clustered", "cluster_entity": True}
    y, x = panel.dependent(), panel.exog()
    mod = PanelOLS(y, x, entity_effects=True)
    expected = mod.fit(clusters=panel.clusters_frame("flat"), **kwargs)
    result = mod.fit(clusters=panel.clusters_frame("flat", "us"), **kwargs)
    assert_same_results(result, expected)


@pytest.mark.parametrize("unit", ["us", "s"])
def test_panel_other_effects(panel, unit):
    y, x = panel.dependent(), panel.exog()
    expected = PanelOLS(y, x, other_effects=panel.other_effects()).fit()
    result = PanelOLS(y, x, other_effects=panel.other_effects(unit)).fit()
    assert_same_results(result, expected)


@pytest.mark.parametrize("unit", ["us", "s"])
def test_panel_formula_with_weights(panel, unit):
    # The variables of a formula share an index, but the weights are separate
    frame = pd.concat([panel.dependent(), panel.exog()], axis=1)
    formula = "y ~ 1 + a + b"
    expected = PooledOLS.from_formula(formula, frame, weights=panel.weights("flat"))
    weights = panel.weights("flat", unit)
    result = PooledOLS.from_formula(formula, frame, weights=weights)
    assert_same_results(result.fit(), expected.fit())
    shifted = panel.weights("flat", unit, shift=1)
    with pytest.raises(ValueError, match="index of weights does not match dependent"):
        PooledOLS.from_formula(formula, frame, weights=shifted)


def test_panel_every_input_has_its_own_resolution(panel):
    kwargs = {"entity_effects": True}
    expected = PanelOLS(
        panel.dependent(),
        panel.exog(),
        weights=panel.weights("flat"),
        other_effects=panel.other_effects(),
        **kwargs,
    ).fit(cov_type="clustered", clusters=panel.clusters_frame("flat"))
    result = PanelOLS(
        panel.dependent("us"),
        panel.exog("ns"),
        weights=panel.weights("flat", "ms"),
        other_effects=panel.other_effects("s"),
        **kwargs,
    ).fit(cov_type="clustered", clusters=panel.clusters_frame("flat", "us"))
    assert_same_results(result, expected)


def test_panel_dependent_without_labels(panel):
    # exog sets the labels, and the weights and clusters have to use them
    y = panel.y.reshape(NENTITY, NTIME).T
    expected = PooledOLS(y, panel.exog(), weights=panel.weights("flat")).fit(
        cov_type="clustered", clusters=panel.clusters_frame("flat")
    )
    exog = panel.exog("us")
    result = PooledOLS(y, exog, weights=panel.weights("flat", "ns")).fit(
        cov_type="clustered", clusters=panel.clusters_frame("flat", "ms")
    )
    assert_same_results(result, expected)
    match = "row index of weights does not match exog"
    with pytest.raises(ValueError, match=match):
        PooledOLS(y, exog, weights=panel.weights("flat", "ns", shift=1))


def test_panel_hausman(panel):
    def hausman(re_unit, fe_unit, shift=0):
        re = RandomEffects(panel.dependent(re_unit), panel.exog(re_unit)).fit()
        fe_y = panel.dependent(fe_unit, shift=shift)
        fe_x = panel.exog(fe_unit, shift=shift)
        fe = PanelOLS(fe_y, fe_x, entity_effects=True).fit()
        return re.hausman(fe)

    expected = hausman("ns", "ns")
    result = hausman("us", "ns")
    assert_allclose(result.statistic.stat, expected.statistic.stat, rtol=1e-10)
    assert_allclose(result.estimates, expected.estimates, rtol=1e-10)
    # Other dates are another sample
    with pytest.raises(ValueError, match="must be estimated using the same obs"):
        hausman("us", "ns", shift=1)


EXOG_CORRUPTIONS = {
    "shifted dates": lambda p: p.exog("us", shift=1),
    "reversed": lambda p: p.exog("us").iloc[::-1],
    "other entities": lambda p: p.exog("us", entities=OTHER_ENTITIES),
}


@pytest.mark.parametrize(("model", "options"), PANEL_MODELS)
@pytest.mark.parametrize("corruption", EXOG_CORRUPTIONS)
def test_panel_other_dates_in_exog_are_rejected(panel, model, options, corruption):
    exog = EXOG_CORRUPTIONS[corruption](panel)
    match = "row index of exog does not match dependent"
    for unit in ("ns", "us"):
        with pytest.raises(ValueError, match=match):
            model(panel.dependent(unit), exog, **options)


WEIGHT_CORRUPTIONS = {
    "flat shifted": lambda p: p.weights("flat", "us", shift=1),
    "flat reversed": lambda p: p.weights("flat", "us").iloc[::-1],
    "grid shifted": lambda p: p.weights("grid", "us", shift=1),
    "grid reversed": lambda p: p.weights("grid", "us").iloc[::-1],
    "time shifted": lambda p: p.weights("time", "us", shift=1),
    "time reversed": lambda p: p.weights("time", "us").iloc[::-1],
}


@pytest.mark.parametrize(("model", "options"), PANEL_MODELS)
@pytest.mark.parametrize("corruption", WEIGHT_CORRUPTIONS)
def test_panel_other_dates_in_weights_are_rejected(panel, model, options, corruption):
    weights = WEIGHT_CORRUPTIONS[corruption](panel)
    with pytest.raises(ValueError, match="index of weights does not match dependent"):
        model(panel.dependent(), panel.exog(), weights=weights, **options)


CLUSTER_CORRUPTIONS = {
    "flat shifted": lambda p: p.clusters_frame("flat", "us", shift=1),
    "flat reversed": lambda p: p.clusters_frame("flat", "us").iloc[::-1],
    "grid shifted": lambda p: p.clusters_frame("grid", "us", shift=1),
    "grid reversed": lambda p: p.clusters_frame("grid", "us").iloc[::-1],
}


@pytest.mark.parametrize(("model", "options"), CLUSTER_MODELS)
@pytest.mark.parametrize("corruption", CLUSTER_CORRUPTIONS)
def test_panel_other_dates_in_clusters_are_rejected(panel, model, options, corruption):
    clusters = CLUSTER_CORRUPTIONS[corruption](panel)
    mod = model(panel.dependent(), panel.exog(), **options)
    with pytest.raises(ValueError, match="index of clusters does not match dependent"):
        mod.fit(cov_type="clustered", clusters=clusters)


@pytest.mark.parametrize("shift", [1, 366])
def test_panel_other_dates_in_other_effects_are_rejected(panel, shift):
    effects = panel.other_effects("us", shift)
    match = "row index of other_effects does not match dependent"
    with pytest.raises(ValueError, match=match):
        PanelOLS(panel.dependent(), panel.exog(), other_effects=effects)


@needs_units
@pytest.mark.parametrize("year", [3000, 1000])
@pytest.mark.parametrize(("model", "options"), PANEL_MODELS)
@pytest.mark.parametrize(("y_unit", "x_unit"), [("s", "us"), ("us", "s"), ("ms", "s")])
def test_panel_dates_outside_the_nanosecond_range(
    panel, year, model, options, y_unit, x_unit
):
    def build(unit_y, unit_x):
        y = pd.Series(panel.y, index=far_index(unit_y, year, NTIME, ENTITIES))
        index = far_index(unit_x, year, NTIME, ENTITIES)
        return y, pd.DataFrame(panel.x, index=index, columns=["a", "b"])

    expected = fit_panel(model, options, *build("us", "us"))
    y, x = build(y_unit, x_unit)
    assert_same_results(fit_panel(model, options, y, x), expected)


@needs_units
@pytest.mark.parametrize("year", [3000, 1000])
def test_panel_dates_outside_the_nanosecond_range_are_not_nanoseconds(panel, year):
    # Nanoseconds cannot hold these dates, and so it has other dates. pandas 3
    # raises its own OutOfBoundsDatetime error from MultiIndex.equals otherwise
    y = pd.Series(panel.y, index=far_index("s", year, NTIME, ENTITIES))
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        PooledOLS(y, panel.exog("ns"))


@pytest.mark.parametrize("data_array", ["dependent", "exog"])
@pytest.mark.parametrize(("y_unit", "x_unit"), UNIT_PAIRS)
def test_panel_two_dimensional_data_arrays(panel, data_array, y_unit, x_unit):
    xr = pytest.importorskip("xarray")
    y = panel.dependent().unstack(0)
    x = panel.exog()["a"].unstack(0)
    expected = PooledOLS(y, x).fit()
    y, x = y.set_axis(dates(y_unit, NTIME)), x.set_axis(dates(x_unit, NTIME))
    if data_array == "dependent":
        coords = {"time": y.index.to_numpy(), "entity": ENTITIES}
        with data_array_warns(y_unit):
            y = xr.DataArray(y.to_numpy(), dims=["time", "entity"], coords=coords)
    else:
        coords = {"time": x.index.to_numpy(), "entity": ENTITIES}
        with data_array_warns(x_unit):
            x = xr.DataArray(x.to_numpy(), dims=["time", "entity"], coords=coords)
    assert_allclose(PooledOLS(y, x).fit().params, expected.params)


# Instrumental variable models

IV_ESTIMATORS = [IV2SLS, IVLIML, IVGMM, IVGMMCUE, IVJIVE]
IV_IDS = [cls.__name__ for cls in IV_ESTIMATORS]
GMM_ESTIMATORS = [IVGMM, IVGMMCUE]
IV_UNITS = [
    {
        "dependent": "us",
        "exog": "ns",
        "endog": "ms",
        "instruments": "s",
        "weights": "us",
    },
    {
        "dependent": "ns",
        "exog": "us",
        "endog": "s",
        "instruments": "ms",
        "weights": "s",
    },
]
IV_UNIT_IDS = ["dependent-us", "dependent-ns"]
ROW_KINDS = ["date", "utc", "timedelta"]


def rows(kind, unit, n, shift=0):
    """Labels of the rows of an input: dates, dates with a time zone or deltas"""
    if kind == "date":
        return dates(unit, n, shift)
    if kind == "utc":
        return dates(unit, n, shift, tz="UTC")
    return deltas(unit, n, shift)


class IVInputs:
    """The values of a model, where the rows of each input have a resolution"""

    nobs = 60

    def __init__(self):
        rng = np.random.default_rng(20261014)
        n = self.nobs
        self.instruments = rng.standard_normal((n, 2))
        self.endog = (self.instruments.sum(axis=1) + rng.standard_normal(n))[:, None]
        self.exog = np.c_[np.ones(n), rng.standard_normal(n)]
        noise = rng.standard_normal(n)
        self.dependent = 1 + self.endog[:, 0] + self.exog[:, 1] + noise
        self.weights = np.exp(rng.standard_normal(n))
        self.clusters = rng.integers(0, 10, n)
        self.absorb = rng.integers(0, 5, n)
        self.cat = rng.integers(0, 3, n)
        self.cont = rng.standard_normal(n)

    def inputs(self, units=None, kind="date", shifted=()):
        """The inputs. A name that is not in units has nanoseconds"""
        units = {} if units is None else units

        def index(name):
            shift = 1 if name in shifted else 0
            return rows(kind, units.get(name, "ns"), self.nobs, shift)

        return {
            "dependent": pd.Series(self.dependent, index=index("dependent"), name="y"),
            "exog": pd.DataFrame(
                self.exog, index=index("exog"), columns=["const", "w"]
            ),
            "endog": pd.DataFrame(self.endog, index=index("endog"), columns=["x"]),
            "instruments": pd.DataFrame(
                self.instruments, index=index("instruments"), columns=["z0", "z1"]
            ),
            "weights": pd.Series(self.weights, index=index("weights"), name="wt"),
            "clusters": pd.Series(self.clusters, index=index("clusters"), name="cl"),
        }


@pytest.fixture(scope="module")
def iv():
    return IVInputs()


def fit_iv(cls, inputs, **weight_config):
    args = [inputs[name] for name in ("dependent", "exog", "endog", "instruments")]
    return cls(*args, weights=inputs["weights"], **weight_config).fit()


@pytest.mark.parametrize("cls", IV_ESTIMATORS, ids=IV_IDS)
@pytest.mark.parametrize("units", IV_UNITS, ids=IV_UNIT_IDS)
def test_iv_inputs(iv, cls, units):
    expected = fit_iv(cls, iv.inputs())
    result = fit_iv(cls, iv.inputs(units))
    assert_same_results(result, expected)


@pytest.mark.parametrize("kind", ROW_KINDS)
@pytest.mark.parametrize("units", IV_UNITS, ids=IV_UNIT_IDS)
def test_iv_rows_with_time_zones_and_time_deltas(iv, kind, units):
    expected = fit_iv(IV2SLS, iv.inputs(kind=kind))
    result = fit_iv(IV2SLS, iv.inputs(units, kind=kind))
    assert_same_results(result, expected)


def test_iv2sls_inputs_match_two_stage_least_squares(iv):
    result = fit_iv(IV2SLS, iv.inputs(IV_UNITS[0]))
    x, z = np.c_[iv.exog, iv.endog], np.c_[iv.exog, iv.instruments]
    root_w = np.sqrt(iv.weights)
    xw, zw, yw = root_w[:, None] * x, root_w[:, None] * z, root_w * iv.dependent
    projection = zw @ np.linalg.solve(zw.T @ zw, zw.T)
    expected = np.linalg.solve(xw.T @ projection @ xw, xw.T @ projection @ yw)
    assert_allclose(result.params, expected)


def fit_with_clusters(cls, inputs):
    args = [inputs["dependent"], inputs["exog"]]
    if cls is not OLS:
        args += [inputs["endog"], inputs["instruments"]]
    mod = cls(*args, weights=inputs["weights"])
    return mod.fit(cov_type="clustered", clusters=inputs["clusters"])


@pytest.mark.parametrize("cls", [*IV_ESTIMATORS, OLS], ids=[*IV_IDS, "OLS"])
@pytest.mark.parametrize("unit", ["us", "s"])
def test_iv_clusters_in_fit(iv, cls, unit):
    expected = fit_with_clusters(cls, iv.inputs())
    result = fit_with_clusters(cls, iv.inputs({"clusters": unit, "weights": "ms"}))
    assert_same_results(result, expected)


@pytest.mark.parametrize("cls", GMM_ESTIMATORS)
@pytest.mark.parametrize("unit", ["us", "s"])
def test_iv_clusters_of_the_weight_estimator(iv, cls, unit):
    def fit(units):
        inputs = iv.inputs(units)
        kwargs = {"weight_type": "clustered", "clusters": inputs["clusters"]}
        return fit_iv(cls, inputs, **kwargs)

    assert_same_results(fit({"clusters": unit}), fit({}))


def test_iv_dependent_without_labels(iv):
    # exog sets the labels, and the other inputs have to use them
    def fit(inputs):
        args = [iv.dependent, *(inputs[k] for k in ("exog", "endog", "instruments"))]
        mod = IV2SLS(*args, weights=inputs["weights"])
        return mod.fit(cov_type="clustered", clusters=inputs["clusters"])

    units = {"exog": "us", "endog": "ns", "instruments": "ms", "weights": "s"}
    assert_same_results(fit(iv.inputs(units)), fit(iv.inputs()))
    inputs = iv.inputs(units | {"clusters": "us"})
    shifted = iv.inputs({"instruments": "us"}, shifted=["instruments"])
    inputs["instruments"] = shifted["instruments"]
    with pytest.raises(
        ValueError, match="row index of instruments does not match exog"
    ):
        fit(inputs)


IV_CORRUPTIONS = {
    "shifted dates": lambda iv, name: iv.inputs({name: "us"}, shifted=[name])[name],
    "reversed": lambda iv, name: iv.inputs({name: "us"})[name].iloc[::-1],
}


@pytest.mark.parametrize("cls", IV_ESTIMATORS, ids=IV_IDS)
@pytest.mark.parametrize("name", ["exog", "endog", "instruments", "weights"])
@pytest.mark.parametrize("corruption", IV_CORRUPTIONS)
def test_iv_other_dates_are_rejected(iv, cls, name, corruption):
    inputs = iv.inputs()
    inputs[name] = IV_CORRUPTIONS[corruption](iv, name)
    match = f"row index of {name} does not match dependent"
    with pytest.raises(ValueError, match=match):
        fit_iv(cls, inputs)


@pytest.mark.parametrize("cls", [IV2SLS, IVGMM, OLS])
@pytest.mark.parametrize("corruption", IV_CORRUPTIONS)
def test_iv_other_dates_in_clusters_are_rejected(iv, cls, corruption):
    inputs = iv.inputs()
    inputs["clusters"] = IV_CORRUPTIONS[corruption](iv, "clusters")
    match = "row index of clusters does not match dependent"
    with pytest.raises(ValueError, match=match):
        fit_with_clusters(cls, inputs)
    if cls is IVGMM:
        kwargs = {"weight_type": "clustered", "clusters": inputs["clusters"]}
        with pytest.raises(ValueError, match=match):
            fit_iv(cls, inputs, **kwargs)


@pytest.mark.parametrize("kind", ["utc", "timedelta"])
def test_iv_other_values_of_time_zones_and_time_deltas_are_rejected(iv, kind):
    inputs = iv.inputs(kind=kind)
    inputs["exog"] = iv.inputs({"exog": "us"}, kind=kind, shifted=["exog"])["exog"]
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        fit_iv(IV2SLS, inputs)


@pytest.mark.parametrize("kind", ["utc", "timedelta"])
def test_iv_dates_time_zones_and_time_deltas_are_different_labels(iv, kind):
    inputs = iv.inputs()
    inputs["exog"] = iv.inputs({"exog": "us"}, kind=kind)["exog"]
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        fit_iv(IV2SLS, inputs)


@pytest.mark.parametrize("cls", [IV2SLS, IVGMM], ids=["IV2SLS", "IVGMM"])
@pytest.mark.parametrize("unit", ["us", "s"])
def test_iv_formula_with_weights(iv, cls, unit):
    # The variables of a formula share an index, but the weights are separate
    inputs = iv.inputs()
    parts = [inputs[k] for k in ("dependent", "endog", "instruments")]
    frame = pd.concat([*parts, inputs["exog"][["w"]]], axis=1)
    formula = "y ~ 1 + w + [x ~ z0 + z1]"
    expected = cls.from_formula(formula, frame, weights=inputs["weights"]).fit()
    weights = iv.inputs({"weights": unit})["weights"]
    result = cls.from_formula(formula, frame, weights=weights).fit()
    assert_same_results(result, expected)
    shifted = iv.inputs({"weights": unit}, shifted=["weights"])["weights"]
    with pytest.raises(ValueError, match="row index of weights does not match"):
        cls.from_formula(formula, frame, weights=shifted)


@needs_units
@pytest.mark.parametrize("year", [3000, 1000])
def test_iv_dates_outside_the_nanosecond_range(iv, year):
    def fit(units):
        n = iv.nobs
        y, exog, endog, instruments = (far_index(unit, year, n) for unit in units)
        mod = IV2SLS(
            pd.Series(iv.dependent, index=y),
            pd.DataFrame(iv.exog, index=exog, columns=["const", "w"]),
            pd.DataFrame(iv.endog, index=endog, columns=["x"]),
            pd.DataFrame(iv.instruments, index=instruments, columns=["z0", "z1"]),
        )
        return mod.fit()

    expected = fit(["us", "us", "us", "us"])
    assert_same_results(fit(["s", "us", "s", "ms"]), expected)
    inputs = iv.inputs()
    far_y = pd.Series(iv.dependent, index=far_index("s", year, iv.nobs))
    inputs["dependent"] = far_y
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        fit_iv(IV2SLS, inputs)


def data_array(frame, unit):
    """DataArray with the rows of the frame, where the coordinates have the unit"""
    xr = pytest.importorskip("xarray")
    obs = frame.index.to_numpy().astype(f"datetime64[{unit}]")
    coords = {"obs": obs, "var": list(frame.columns)}
    with data_array_warns(unit):
        return xr.DataArray(frame.to_numpy(), dims=["obs", "var"], coords=coords)


@pytest.mark.parametrize("unit", ["us", "s", "ns"])
@pytest.mark.parametrize("y_unit", ["us", "ns"])
def test_iv_data_array_and_pandas_inputs(iv, unit, y_unit):
    expected = fit_iv(IV2SLS, iv.inputs())
    inputs = iv.inputs({"dependent": y_unit})
    inputs["exog"] = data_array(iv.inputs()["exog"], unit)
    assert_same_results(fit_iv(IV2SLS, inputs), expected)


@pytest.mark.parametrize("unit", ["us", "s", "ns"])
def test_iv_data_array_with_other_dates_is_rejected(iv, unit):
    inputs = iv.inputs()
    shifted = iv.inputs({"exog": unit}, shifted=["exog"])["exog"]
    inputs["exog"] = data_array(shifted, unit)
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        fit_iv(IV2SLS, inputs)


# AbsorbingLS


def absorbing_inputs(iv, units=None, shifted=()):
    """The inputs of AbsorbingLS. A name that is not in units has nanoseconds"""
    units = {} if units is None else units

    def index(name):
        shift = 1 if name in shifted else 0
        return dates(units.get(name, "ns"), iv.nobs, shift)

    return {
        "dependent": pd.Series(iv.dependent, index=index("dependent"), name="y"),
        "exog": pd.DataFrame(iv.exog, index=index("exog"), columns=["const", "w"]),
        "absorb": pd.DataFrame({"g": pd.Categorical(iv.absorb)}, index=index("absorb")),
        "cat": pd.DataFrame({"h": pd.Categorical(iv.cat)}, index=index("cat")),
        "cont": pd.DataFrame({"c": iv.cont}, index=index("cont")),
        "weights": pd.Series(iv.weights, index=index("weights"), name="wt"),
        "clusters": pd.Series(iv.clusters, index=index("clusters"), name="cl"),
    }


def absorbing_model(inputs, interactions=None):
    if interactions is None:
        interactions = Interaction(cat=inputs["cat"], cont=inputs["cont"])
    return AbsorbingLS(
        inputs["dependent"],
        inputs["exog"],
        absorb=inputs["absorb"],
        interactions=interactions,
        weights=inputs["weights"],
    )


def test_absorbing_inputs(iv):
    units = {
        "dependent": "us",
        "exog": "ms",
        "absorb": "s",
        "cat": "us",
        "cont": "ms",
        "weights": "us",
        "clusters": "s",
    }
    kwargs = {"cov_type": "clustered"}
    expected = absorbing_model(absorbing_inputs(iv)).fit(
        clusters=absorbing_inputs(iv)["clusters"], **kwargs
    )
    inputs = absorbing_inputs(iv, units)
    result = absorbing_model(inputs).fit(clusters=inputs["clusters"], **kwargs)
    assert_same_results(result, expected)


@pytest.mark.parametrize("name", ["exog", "absorb", "weights"])
def test_absorbing_other_dates_are_rejected(iv, name):
    inputs = absorbing_inputs(iv)
    inputs[name] = absorbing_inputs(iv, {name: "us"}, shifted=[name])[name]
    match = f"row index of {name} does not match dependent"
    with pytest.raises(ValueError, match=match):
        absorbing_model(inputs)


def test_absorbing_other_dates_in_interactions_are_rejected(iv):
    shifted = absorbing_inputs(iv, {"cat": "us", "cont": "ms"}, ["cat", "cont"])
    interactions = Interaction(cat=shifted["cat"], cont=shifted["cont"])
    match = "row index of interactions does not match dependent"
    with pytest.raises(ValueError, match=match):
        absorbing_model(absorbing_inputs(iv), interactions)
    # The two parts of an interaction have to use the same labels
    inputs = absorbing_inputs(iv, {"cat": "us"})
    shifted_cont = absorbing_inputs(iv, {"cont": "ms"}, ["cont"])["cont"]
    match = "row index of cont does not match cat"
    with pytest.raises(ValueError, match=match):
        Interaction(cat=inputs["cat"], cont=shifted_cont)


# System models

SYSTEM_MODELS = [
    pytest.param(SUR, "sur", id="SUR"),
    pytest.param(IV3SLS, "iv", id="IV3SLS"),
    pytest.param(IVSystemGMM, "iv", id="IVSystemGMM"),
]
SYSTEM_UNITS = [
    {("a", "dependent"): "us"},
    {("b", "dependent"): "us", ("b", "exog"): "ns"},
    {
        ("a", "dependent"): "us",
        ("a", "exog"): "ms",
        ("a", "weights"): "s",
        ("b", "dependent"): "ns",
        ("b", "exog"): "us",
        ("b", "weights"): "ms",
    },
]
SYSTEM_UNIT_IDS = ["first-dependent", "second-dependent", "everything"]


class SystemInputs:
    """The values of a system, where the rows of each input have a resolution"""

    nobs = 60

    def __init__(self):
        rng = np.random.default_rng(20261015)
        n = self.nobs
        common = rng.standard_normal(n)
        self.values = {}
        for label in ("a", "b"):
            instruments = rng.standard_normal((n, 2))
            endog = (instruments.sum(axis=1) + rng.standard_normal(n))[:, None]
            exog = np.c_[np.ones(n), rng.standard_normal(n)]
            noise = common + rng.standard_normal(n)
            self.values[label] = {
                "dependent": 1 + exog[:, 1] + endog[:, 0] + noise,
                "exog": exog,
                "endog": endog,
                "instruments": instruments,
                "weights": np.exp(rng.standard_normal(n)),
            }
        self.clusters = rng.integers(0, 8, n)

    def equations(self, kind, units=None, shifted=()):
        """
        The equations, as dictionaries of pandas objects

        A (equation, input) that is not in units has nanoseconds, and the dates
        of the ones in shifted start one day later.
        """
        units = {} if units is None else units
        equations = {}
        for label, values in self.values.items():
            equation = {}
            for name, value in values.items():
                if kind == "sur" and name in ("endog", "instruments"):
                    continue
                shift = 1 if (label, name) in shifted else 0
                index = dates(units.get((label, name), "ns"), self.nobs, shift)
                if value.ndim == 1:
                    equation[name] = pd.Series(
                        value, index=index, name=f"{name}_{label}"
                    )
                else:
                    columns = [f"{name}_{label}{i}" for i in range(value.shape[1])]
                    equation[name] = pd.DataFrame(value, index=index, columns=columns)
            equations[f"eq_{label}"] = equation
        return equations


@pytest.fixture(scope="module")
def system():
    return SystemInputs()


@pytest.mark.parametrize(("cls", "kind"), SYSTEM_MODELS)
@pytest.mark.parametrize("units", SYSTEM_UNITS, ids=SYSTEM_UNIT_IDS)
def test_system_inputs(system, cls, kind, units):
    expected = cls(system.equations(kind)).fit()
    result = cls(system.equations(kind, units)).fit()
    assert_same_results(result, expected, resids=False)


@pytest.mark.parametrize(("cls", "kind"), SYSTEM_MODELS[:2])
@pytest.mark.parametrize("unit", ["us", "s"])
def test_system_clusters(system, cls, kind, unit):
    mod = cls(system.equations(kind))
    clusters = pd.Series(system.clusters, index=dates("ns", system.nobs))
    expected = mod.fit(cov_type="clustered", clusters=clusters)
    clusters = pd.Series(system.clusters, index=dates(unit, system.nobs))
    result = mod.fit(cov_type="clustered", clusters=clusters)
    assert_same_results(result, expected, resids=False)


@pytest.mark.parametrize(("cls", "kind"), SYSTEM_MODELS)
@pytest.mark.parametrize("label", ["a", "b"])
def test_system_other_dates_in_the_dependent_are_rejected(system, cls, kind, label):
    equations = system.equations(
        kind, {(label, "dependent"): "us"}, shifted=[(label, "dependent")]
    )
    match = (
        "row index of dependent in equation 'eq_b' does not match "
        "dependent in equation 'eq_a'"
    )
    with pytest.raises(ValueError, match=match):
        cls(equations)


@pytest.mark.parametrize(("cls", "kind"), SYSTEM_MODELS)
@pytest.mark.parametrize("name", ["exog", "weights"])
@pytest.mark.parametrize("label", ["a", "b"])
def test_system_other_dates_in_the_inputs_are_rejected(system, cls, kind, name, label):
    equations = system.equations(kind, {(label, name): "us"}, shifted=[(label, name)])
    match = (
        f"row index of {name} in equation 'eq_{label}' does not match "
        f"dependent in equation 'eq_{label}'"
    )
    with pytest.raises(ValueError, match=match):
        cls(equations)


# Asset pricing models

ASSET_MODELS = [TradedFactorModel, LinearFactorModel, LinearFactorModelGMM]
ASSET_IDS = [cls.__name__ for cls in ASSET_MODELS]


class ReturnsInputs:
    """The values of the returns, where the rows of each input have a resolution"""

    nobs = 100

    def __init__(self):
        rng = np.random.default_rng(20261016)
        self.factors = 0.5 + rng.standard_normal((self.nobs, 2))
        loadings = rng.chisquare(2, (2, 5)) / 2
        self.portfolios = self.factors @ loadings + rng.standard_normal((self.nobs, 5))

    def portfolio_frame(self, unit="ns", shift=0):
        columns = [f"p{i}" for i in range(5)]
        index = dates(unit, self.nobs, shift)
        return pd.DataFrame(self.portfolios, index=index, columns=columns)

    def factor_frame(self, unit="ns", shift=0):
        index = dates(unit, self.nobs, shift)
        return pd.DataFrame(self.factors, index=index, columns=["f0", "f1"])


@pytest.fixture(scope="module")
def returns():
    return ReturnsInputs()


def fit_asset(cls, portfolios, factors):
    mod = cls(portfolios, factors)
    if cls is LinearFactorModelGMM:
        return mod.fit(disp=0)
    return mod.fit()


# LinearFactorModelGMM minimizes the GMM objective with BFGS, whose gradient is a finite
# difference. Fits of the same data that only differ by rounding stop at estimates that
# differ by up to 1e-7 (4e-8 on Intel macOS), and not by 1e-16.
ASSET_RTOL = {LinearFactorModelGMM: 1e-6}


@pytest.mark.parametrize("cls", ASSET_MODELS, ids=ASSET_IDS)
@pytest.mark.parametrize(("p_unit", "f_unit"), UNIT_PAIRS)
def test_asset_pricing_inputs(returns, cls, p_unit, f_unit):
    expected = fit_asset(cls, returns.portfolio_frame(), returns.factor_frame())
    portfolios, factors = returns.portfolio_frame(p_unit), returns.factor_frame(f_unit)
    result = fit_asset(cls, portfolios, factors)
    rtol = ASSET_RTOL.get(cls, 1e-10)
    assert_same_results(result, expected, resids=False, rtol=rtol)


@pytest.mark.parametrize("cls", ASSET_MODELS, ids=ASSET_IDS)
@pytest.mark.parametrize("corruption", ["shifted dates", "reversed"])
def test_asset_pricing_other_dates_are_rejected(returns, cls, corruption):
    if corruption == "reversed":
        factors = returns.factor_frame("us").iloc[::-1]
    else:
        factors = returns.factor_frame("us", shift=1)
    with pytest.raises(ValueError, match="row index of factors does not match"):
        cls(returns.portfolio_frame(), factors)
