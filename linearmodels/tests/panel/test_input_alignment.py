"""
Inputs that carry labels (pandas, xarray) have to use the labels of dependent.

Estimation pairs rows by position, so a labelled input whose rows are in a
different order from dependent would silently be attached to the wrong
observations. These tests check that this is rejected, that correctly
labelled inputs are accepted for every supported shape of weights, and that
raw arrays continue to be paired by position. If dependent carries no labels,
the first of exog, weights and other_effects that does sets the labels, and the
inputs after it, including the clusters, have to use them.
"""

from linearmodels.compat.pandas import ANNUAL_FREQ

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import (
    AmbiguityError,
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from linearmodels.shared.exceptions import MissingValueWarning

NENTITY = 8
NTIME = 5
WEIGHT_SHAPES = ["time", "entity", "grid", "flat"]
ROW_ORDERS = ["entity-major", "time-major"]
LABEL_STYLES = ["str-date", "int-int"]
MODELS = [
    PooledOLS,
    PanelOLS,
    RandomEffects,
    BetweenOLS,
    FirstDifferenceOLS,
    FamaMacBeth,
]
MODEL_IDS = [m.__name__ for m in MODELS]


def make_labels(style):
    if style == "str-date":
        entities = pd.Index([f"firm{i}" for i in range(NENTITY)])
        times = pd.date_range("2001-12-31", periods=NTIME, freq=ANNUAL_FREQ)
    else:
        entities = pd.Index(np.arange(100, 100 + NENTITY))
        times = pd.Index(np.arange(1990, 1990 + NTIME))
    return entities, times


class Panel:
    """Labelled panel in either entity-major or time-major row order"""

    def __init__(self, style, order):
        rng = np.random.default_rng(20240607)
        self.entities, self.times = make_labels(style)
        entity_major = pd.MultiIndex.from_product(
            [self.entities, self.times], names=["entity", "time"]
        )
        n = len(entity_major)
        x = pd.DataFrame(
            rng.standard_normal((n, 2)), index=entity_major, columns=["x0", "x1"]
        )
        beta = np.array([0.5, -1.0])
        y = pd.Series(
            x.to_numpy() @ beta + rng.standard_normal(n), index=entity_major, name="y"
        )
        # Positive weights for every (entity, time) cell and in every shape
        self.w_grid = pd.DataFrame(
            rng.chisquare(5, (NTIME, NENTITY)) + 0.5,
            index=self.times,
            columns=self.entities,
        )
        self.w_time = pd.Series(
            rng.chisquare(5, NTIME) + 0.5, index=self.times, name="w"
        )
        self.w_entity = pd.Series(
            rng.chisquare(5, NENTITY) + 0.5, index=self.entities, name="w"
        )
        self.w_flat = pd.Series(rng.chisquare(5, n) + 0.5, index=entity_major, name="w")
        cats = pd.DataFrame({"c": np.arange(n) % 3}, index=entity_major, dtype=np.int64)
        self.clusters = pd.DataFrame(
            {"cl": np.arange(n) % 4}, index=entity_major, dtype=np.int64
        )
        if order == "time-major":
            index = pd.MultiIndex.from_tuples(
                sorted(entity_major, key=lambda p: (p[1], p[0])),
                names=entity_major.names,
            )
            x, y, cats = x.reindex(index), y.reindex(index), cats.reindex(index)
            self.clusters = self.clusters.reindex(index)
            self.w_flat = self.w_flat.reindex(index)
        self.x, self.y, self.other = x, y, cats
        self.index = y.index

    def expected_weights(self, shape):
        """Label based weights of every row in the order of dependent"""
        if shape == "time":
            full = self.w_time.reindex(self.index.get_level_values(1))
            full.index = self.index
            return full
        if shape == "entity":
            full = self.w_entity.reindex(self.index.get_level_values(0))
            full.index = self.index
            return full
        if shape == "grid":
            stacked = self.w_grid.T.stack()
            stacked.index.names = self.index.names
            return stacked.reindex(self.index)
        return self.w_flat

    def labelled_weights(self, shape):
        """Weights in the form the user would pass, with labels"""
        if shape == "time":
            return self.w_time
        if shape == "entity":
            return self.w_entity
        if shape == "grid":
            return self.w_grid
        return self.w_flat

    def raw_weights(self, shape):
        """Weights as an array, paired by position in order of appearance"""
        if shape == "time":
            return self.w_time.to_numpy()
        if shape == "entity":
            return self.w_entity.to_numpy()
        if shape == "grid":
            return self.w_grid.to_numpy()
        return self.w_flat.to_numpy()


@pytest.fixture(params=ROW_ORDERS)
def order(request):
    return request.param


@pytest.fixture(params=LABEL_STYLES)
def style(request):
    return request.param


@pytest.fixture
def panel(style, order):
    return Panel(style, order)


@pytest.fixture
def simple():
    return Panel("str-date", "entity-major")


def to_3d(frame):
    """(variable, time, entity) array from an entity-major frame"""
    values = frame.to_numpy().reshape(NENTITY, NTIME, -1)
    return values.transpose(2, 1, 0)


def pooled_wls(y, x, w):
    root_w = np.sqrt(np.asarray(w, dtype=float))[:, None]
    params, *_ = np.linalg.lstsq(
        root_w * x.to_numpy(), root_w[:, 0] * y.to_numpy(), rcond=None
    )
    return params


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_exog_reordered_rejected(simple, model):
    model(simple.y, simple.x)
    with pytest.raises(ValueError, match="row index of exog"):
        model(simple.y, simple.x.iloc[::-1])
    with pytest.raises(ValueError, match="row index of exog"):
        model(simple.y.iloc[::-1], simple.x)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_exog_misaligned_labels_rejected(simple, model):
    with pytest.raises(ValueError, match="row index of exog"):
        model(simple.y, shift_labels(simple.x))


@pytest.mark.parametrize(
    "model", [PooledOLS, PanelOLS, RandomEffects, BetweenOLS, FamaMacBeth]
)
def test_reordering_everything_is_allowed(simple, model):
    expected = model(simple.y, simple.x).fit().params
    reordered = model(simple.y.iloc[::-1], simple.x.iloc[::-1]).fit().params
    assert_allclose(reordered, expected)


def test_reordering_everything_is_allowed_fd(simple):
    expected = FirstDifferenceOLS(simple.y, simple.x).fit().params
    reordered = FirstDifferenceOLS(simple.y.iloc[::-1], simple.x.iloc[::-1])
    assert_allclose(reordered.fit().params, expected)


def test_exog_raw_array_is_positional(simple):
    expected = PooledOLS(simple.y, simple.x).fit().params
    res = PooledOLS(simple.y, to_3d(simple.x)).fit()
    assert_allclose(res.params, expected)
    # A raw array is paired by position, even when its order differs
    raw = simple.x.to_numpy()[::-1][:, :1]
    res = PooledOLS(simple.y, raw).fit()
    positional, *_ = np.linalg.lstsq(raw, simple.y.to_numpy(), rcond=None)
    assert_allclose(res.params, positional)


def test_unlabelled_dependent_is_paired_by_position(simple):
    # There are no labels in dependent, so exog is paired by position
    y = unlabelled_dependent(simple)
    x = simple.x.iloc[::-1]
    res = PooledOLS(y, x).fit()
    positional, *_ = np.linalg.lstsq(x.to_numpy(), simple.y.to_numpy(), rcond=None)
    assert_allclose(res.params, positional)


def unlabelled_dependent(panel):
    """A time by entity array with the rows of an entity-major dependent"""
    return panel.y.to_numpy().reshape(NENTITY, NTIME).T


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
def test_weights_must_use_the_labels_of_exog_without_labels_in_dependent(
    simple, model, shape
):
    y = unlabelled_dependent(simple)
    weights = simple.labelled_weights(shape)
    mod = model(y, simple.x, weights=weights)
    raw = model(y, simple.x, weights=simple.raw_weights(shape))
    assert_allclose(mod.weights.values2d, raw.weights.values2d)
    for corruption in CORRUPTIONS[shape]:
        bad = CORRUPTIONS[shape][corruption](weights)
        with pytest.raises(ValueError, match=r"index of weights does not match exog"):
            model(y, simple.x, weights=bad)


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_exog_with_labels_must_agree_with_dependent_labels(simple, model):
    # If dependent has labels they are still the ones that are used
    with pytest.raises(ValueError, match="row index of exog does not match dependent"):
        model(simple.y, simple.x.iloc[::-1])
    with pytest.raises(
        ValueError, match="row index of weights does not match dependent"
    ):
        model(simple.y, simple.x, weights=simple.w_flat.iloc[::-1])


def test_message_names_the_input_that_sets_the_labels(simple):
    y = unlabelled_dependent(simple)
    with pytest.raises(ValueError, match="does not match") as exc:
        PooledOLS(y, simple.x, weights=simple.w_flat.iloc[::-1])
    message = str(exc.value)
    assert message.startswith("The row index of weights does not match exog.")
    assert "in the same order as exog" in message
    assert "reindex weights to the index of exog" in message
    assert ".to_numpy()" in message


@pytest.mark.parametrize("shape", ["grid", "flat"])
def test_weights_set_the_labels_without_labels_in_dependent_and_exog(simple, shape):
    y, x = unlabelled_dependent(simple), to_3d(simple.x)
    weights = simple.labelled_weights(shape)
    mod = PanelOLS(y, x, weights=weights, other_effects=simple.other)
    raw = PanelOLS(
        y, x, weights=simple.raw_weights(shape), other_effects=to_3d(simple.other)
    )
    assert_allclose(mod.weights.values2d, raw.weights.values2d)
    match = "row index of other_effects does not match weights"
    with pytest.raises(ValueError, match=match):
        PanelOLS(y, x, weights=weights, other_effects=simple.other.iloc[::-1])
    with pytest.raises(ValueError, match=match):
        PanelOLS(y, x, weights=weights, other_effects=shift_labels(simple.other))
    # Weights for each time period or entity do not have the labels of the rows
    for shape_ in ("time", "entity"):
        weights = simple.labelled_weights(shape_)
        PanelOLS(y, x, weights=weights, other_effects=simple.other.iloc[::-1])


def test_other_effects_set_the_labels_without_other_labels(simple):
    y, x = unlabelled_dependent(simple), to_3d(simple.x)
    mod = PanelOLS(y, x, other_effects=simple.other)
    raw = PanelOLS(y, x, other_effects=to_3d(simple.other))
    assert_allclose(mod.fit().params, raw.fit().params)
    mod.fit(cov_type="clustered", clusters=simple.clusters)
    match = "row index of clusters does not match other_effects"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=simple.clusters.iloc[::-1])


def test_other_effects_use_the_labels_of_exog(simple):
    y = unlabelled_dependent(simple)
    PanelOLS(y, simple.x, other_effects=simple.other)
    for corrupt in (reverse, shift_labels):
        with pytest.raises(
            ValueError, match="row index of other_effects does not match exog"
        ):
            PanelOLS(y, simple.x, other_effects=corrupt(simple.other))
    wide = simple.other["c"].unstack(0)
    with pytest.raises(
        ValueError, match="time and entity index of other_effects does not match exog"
    ):
        PanelOLS(y, simple.x, other_effects=wide.iloc[::-1])
    # An array has no labels
    PanelOLS(y, simple.x, other_effects=to_3d(simple.other)[:, ::-1])


@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
def test_weights_labelled_placed_by_label(panel, shape):
    mod = PooledOLS(panel.y, panel.x, weights=panel.labelled_weights(shape))
    expected = panel.expected_weights(shape)
    expected = expected / expected.mean()
    assert mod.weights.dataframe.index.equals(panel.index)
    assert_allclose(mod.weights.values2d[:, 0], expected.to_numpy())


@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
def test_weights_raw_array_placed_by_position(panel, shape):
    mod = PooledOLS(panel.y, panel.x, weights=panel.raw_weights(shape))
    expected = panel.expected_weights(shape)
    expected = expected / expected.mean()
    assert mod.weights.dataframe.index.equals(panel.index)
    assert_allclose(mod.weights.values2d[:, 0], expected.to_numpy())


@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
@pytest.mark.parametrize("labelled", [True, False], ids=["labelled", "raw"])
def test_weights_estimates_match_wls(panel, shape, labelled):
    weights = panel.labelled_weights(shape) if labelled else panel.raw_weights(shape)
    res = PooledOLS(panel.y, panel.x, weights=weights).fit()
    expected = pooled_wls(panel.y, panel.x, panel.expected_weights(shape))
    assert_allclose(res.params.to_numpy(), expected)


@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_weights_shapes_all_models(simple, shape, model):
    labelled = model(simple.y, simple.x, weights=simple.labelled_weights(shape))
    raw = model(simple.y, simple.x, weights=simple.raw_weights(shape))
    assert_allclose(labelled.weights.values2d, raw.weights.values2d)
    assert_allclose(labelled.fit().params, raw.fit().params)


def reverse(value):
    return value.iloc[::-1]


def shift_labels(value):
    """Valid labels, but attached to the wrong rows"""
    out = value.copy()
    if isinstance(out.index, pd.MultiIndex):
        times = out.index.get_level_values(1)
        shifted = dict(zip(times.unique(), np.roll(times.unique(), 1), strict=True))
        out.index = pd.MultiIndex.from_arrays(
            [out.index.get_level_values(0), times.map(shifted)]
        )
    else:
        out.index = np.roll(out.index, 1)
    return out


def reverse_columns(value):
    return value[value.columns[::-1]]


def default_labels(value):
    """Remove all labels, leaving only the position of each value"""
    if isinstance(value, pd.DataFrame):
        return pd.DataFrame(value.to_numpy())
    return pd.Series(value.to_numpy())


CORRUPTIONS = {
    "time": {"reverse": reverse, "relabel": shift_labels},
    "entity": {"reverse": reverse, "relabel": shift_labels},
    "grid": {
        "reverse-rows": reverse,
        "reverse-columns": reverse_columns,
        "reverse-default-labels": lambda w: default_labels(w).iloc[::-1],
    },
    "flat": {"reverse": reverse, "relabel": shift_labels},
}
CASES = [(s, c) for s, options in CORRUPTIONS.items() for c in options]


@pytest.mark.parametrize(("shape", "corruption"), CASES)
def test_weights_misaligned_rejected(panel, shape, corruption):
    weights = CORRUPTIONS[shape][corruption](panel.labelled_weights(shape))
    with pytest.raises(ValueError, match=r"index of weights does not match"):
        PooledOLS(panel.y, panel.x, weights=weights)
    with pytest.raises(ValueError, match=r"index of weights does not match"):
        PanelOLS(panel.y, panel.x, weights=weights, entity_effects=True)


@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
def test_weights_default_index_is_positional(panel, shape):
    # Without labels there is nothing to compare, so the values are paired by
    # position, as for a NumPy array
    weights = default_labels(panel.labelled_weights(shape))
    mod = PooledOLS(panel.y, panel.x, weights=weights)
    raw = PooledOLS(panel.y, panel.x, weights=panel.raw_weights(shape))
    assert_allclose(mod.weights.values2d, raw.weights.values2d)
    expected = panel.expected_weights(shape)
    assert_allclose(mod.weights.values2d[:, 0], expected / expected.mean())


@pytest.mark.parametrize("shape", ["time", "entity", "flat"])
def test_weights_reversed_default_index_rejected(panel, shape):
    weights = default_labels(panel.labelled_weights(shape)).iloc[::-1]
    assert isinstance(weights.index, pd.RangeIndex)
    with pytest.raises(ValueError, match=r"index of weights does not match"):
        PooledOLS(panel.y, panel.x, weights=weights)


def test_exog_default_index_is_positional(simple):
    exog = pd.DataFrame(simple.x.to_numpy()[:, :1])
    res = PooledOLS(simple.y, exog).fit()
    positional, *_ = np.linalg.lstsq(exog.to_numpy(), simple.y.to_numpy(), rcond=None)
    assert_allclose(res.params, positional)
    with pytest.raises(ValueError, match="row index of exog"):
        PooledOLS(simple.y, exog.iloc[::-1])


def test_per_time_weights_follow_time_not_position(simple):
    # Per-time weights have to follow time, not the position of the row
    index = pd.MultiIndex.from_tuples(
        sorted(simple.y.index, key=lambda p: (p[1], p[0])), names=simple.y.index.names
    )
    y, x = simple.y.reindex(index), simple.x.reindex(index)
    weights = np.arange(1.0, NTIME + 1)
    res = PooledOLS(y, x, weights=weights).fit()
    expected = np.asarray(
        index.get_level_values(1).map(dict(zip(simple.times, weights, strict=True)))
    )
    assert_allclose(res.params, pooled_wls(y, x, expected))


def test_weights_ambiguity_not_resolved_by_labels():
    entities, times = make_labels("str-date")
    entities = entities[:NTIME]
    index = pd.MultiIndex.from_product([entities, times])
    rng = np.random.default_rng(0)
    y = pd.Series(rng.standard_normal(len(index)), index=index)
    x = pd.DataFrame(rng.standard_normal((len(index), 1)), index=index)
    weights = pd.Series(np.ones(NTIME), index=times)
    with pytest.raises(AmbiguityError):
        PooledOLS(y, x, weights=weights)


def test_weights_unbalanced_dependent(simple):
    # An observation is missing, so that the first entity does not have the fourth
    # period. The weights of that period and entity are not used. They raised "weights
    # must have the same number of observations as dependent" before
    keep = np.ones(len(simple.y), dtype=bool)
    keep[3] = False
    y, x = simple.y[keep], simple.x[keep]
    for shape in ("time", "entity", "grid"):
        expected = simple.expected_weights(shape)[keep]
        expected = expected / expected.mean()
        for weights in (simple.raw_weights(shape), simple.labelled_weights(shape)):
            mod = PooledOLS(y, x, weights=weights)
            assert_allclose(mod.weights.values2d[:, 0], expected)
    flat = simple.w_flat[keep]
    mod = PooledOLS(y, x, weights=flat)
    assert_allclose(mod.weights.values2d[:, 0], flat / flat.mean())
    with pytest.raises(ValueError, match="row index of weights"):
        PooledOLS(y, x, weights=flat.iloc[::-1])


def test_weights_panel_data(simple):
    PooledOLS(simple.y, simple.x, weights=PanelData(simple.w_flat))
    with pytest.raises(ValueError, match="row index of weights"):
        PooledOLS(simple.y, simple.x, weights=PanelData(simple.w_flat.iloc[::-1]))


def test_weights_incorrect_shape_still_reported(simple):
    with pytest.raises(ValueError, match="Weights do not have a supported shape"):
        PooledOLS(simple.y, simple.x, weights=simple.w_flat.iloc[:-1])
    with pytest.raises(ValueError, match="Weights do not have a supported shape"):
        PooledOLS(simple.y, simple.x, weights=simple.w_grid.iloc[:, :-1])


def test_other_effects_alignment(panel):
    other = panel.other
    mod = PanelOLS(panel.y, panel.x, other_effects=other)
    assert mod.other_effects
    with pytest.raises(ValueError, match="row index of other_effects"):
        PanelOLS(panel.y, panel.x, other_effects=other.iloc[::-1])
    with pytest.raises(ValueError, match="row index of other_effects"):
        PanelOLS(panel.y, panel.x, other_effects=shift_labels(other))
    wide = other["c"].unstack(0)
    with pytest.raises(ValueError, match="time and entity index of other_effects"):
        PanelOLS(panel.y, panel.x, other_effects=wide.iloc[::-1])


def test_other_effects_raw_array_is_positional(simple):
    labelled = PanelOLS(simple.y, simple.x, other_effects=simple.other)
    raw = PanelOLS(simple.y, simple.x, other_effects=to_3d(simple.other))
    assert_allclose(labelled.fit().params, raw.fit().params)


def test_clusters_alignment(panel):
    clusters = panel.clusters
    mod = PanelOLS(panel.y, panel.x)
    assert mod.reformat_clusters(clusters).dataframe.index.equals(panel.index)
    mod.fit(cov_type="clustered", clusters=clusters)
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.fit(cov_type="clustered", clusters=clusters.iloc[::-1])
    with pytest.raises(ValueError, match="row index of clusters"):
        mod.reformat_clusters(shift_labels(clusters))
    wide = clusters["cl"].unstack(0)
    with pytest.raises(ValueError, match="time and entity index of clusters"):
        mod.reformat_clusters(wide.iloc[::-1])


def test_clusters_raw_array_is_positional(simple):
    mod = PanelOLS(simple.y, simple.x)
    labelled = mod.fit(cov_type="clustered", clusters=simple.clusters)
    raw = mod.fit(cov_type="clustered", clusters=to_3d(simple.clusters))
    assert_allclose(labelled.std_errors, raw.std_errors)


def test_clusters_unlabelled_dependent(simple):
    y = simple.y.to_numpy().reshape(NENTITY, NTIME).T
    mod = PooledOLS(y, to_3d(simple.x))
    clusters = simple.clusters.iloc[::-1]
    assert mod.reformat_clusters(clusters).dataframe.shape == (NENTITY * NTIME, 1)


@pytest.mark.parametrize("model", [PooledOLS, PanelOLS, RandomEffects])
def test_clusters_use_the_labels_of_the_first_input_with_labels(simple, model):
    y = unlabelled_dependent(simple)
    mod = model(y, simple.x)
    clusters = simple.clusters
    res = mod.fit(cov_type="clustered", clusters=clusters)
    raw = model(y, to_3d(simple.x)).fit(cov_type="clustered", clusters=to_3d(clusters))
    assert_allclose(res.std_errors, raw.std_errors)
    match = "row index of clusters does not match exog"
    for corrupt in (reverse, shift_labels):
        with pytest.raises(ValueError, match=match):
            mod.fit(cov_type="clustered", clusters=corrupt(clusters))
        with pytest.raises(ValueError, match=match):
            mod.reformat_clusters(corrupt(clusters))
    wide_match = "time and entity index of clusters does not match exog"
    with pytest.raises(ValueError, match=wide_match):
        mod.reformat_clusters(clusters["cl"].unstack(0).iloc[::-1])
    # Without any labels there is nothing to compare
    mod = model(y, to_3d(simple.x))
    mod.fit(cov_type="clustered", clusters=reverse(clusters))
    # weights can be the first input with labels
    mod = model(y, to_3d(simple.x), weights=simple.w_flat)
    mod.fit(cov_type="clustered", clusters=clusters)
    match = "row index of clusters does not match weights"
    with pytest.raises(ValueError, match=match):
        mod.fit(cov_type="clustered", clusters=reverse(clusters))


def test_labelled_clusters_use_the_labels_of_the_rows_of_dependent(simple):
    # The labels of the rows are those of dependent, which are not the labels of
    # the clusters when dependent is an array
    y = unlabelled_dependent(simple)
    mod = PanelOLS(y, simple.x)
    clusters = mod.reformat_clusters(simple.clusters)
    assert clusters.dataframe.index.equals(mod.dependent.index)
    assert not clusters.dataframe.index.equals(simple.clusters.index)
    assert_allclose(clusters.values2d, simple.clusters.to_numpy())
    # Clustering by entity as well as by the clusters used to fail
    res = mod.fit(cov_type="clustered", clusters=simple.clusters, cluster_entity=True)
    raw = PanelOLS(y, to_3d(simple.x)).fit(
        cov_type="clustered", clusters=to_3d(simple.clusters), cluster_entity=True
    )
    assert np.all(np.isfinite(res.std_errors))
    assert_allclose(res.std_errors, raw.std_errors)


def test_labelled_clusters_first_difference_without_labels_in_dependent(simple):
    y = unlabelled_dependent(simple)
    entity_clusters = pd.DataFrame(
        {"cl": np.repeat(np.arange(NENTITY) % 3, NTIME)}, index=simple.index
    )
    mod = FirstDifferenceOLS(y, simple.x)
    res = mod.fit(cov_type="clustered", clusters=entity_clusters)
    raw = FirstDifferenceOLS(y, to_3d(simple.x)).fit(
        cov_type="clustered", clusters=to_3d(entity_clusters)
    )
    assert_allclose(res.std_errors, raw.std_errors)
    with pytest.raises(ValueError, match="row index of clusters does not match exog"):
        mod.fit(cov_type="clustered", clusters=reverse(entity_clusters))


def test_clusters_and_missing_values_without_labels_in_dependent(simple):
    y = unlabelled_dependent(simple)
    x = simple.x.copy()
    x.iloc[5, 0] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = PooledOLS(y, x)
    assert mod.not_null.sum() == len(simple.y) - 1
    res = mod.fit(cov_type="clustered", clusters=simple.clusters)
    with pytest.warns(MissingValueWarning):
        raw_mod = PooledOLS(y, to_3d(x))
    raw = raw_mod.fit(cov_type="clustered", clusters=to_3d(simple.clusters))
    assert_allclose(res.std_errors, raw.std_errors)
    with pytest.raises(ValueError, match="row index of clusters does not match exog"):
        mod.fit(cov_type="clustered", clusters=reverse(simple.clusters))


def test_xarray_alignment():
    xr = pytest.importorskip("xarray")
    simple = Panel("int-int", "entity-major")
    coords = {
        "vars": ["x0", "x1"],
        "time": simple.times,
        "entities": simple.entities,
    }
    values = simple.x.to_numpy().T.reshape(2, NENTITY, NTIME).transpose(0, 2, 1)
    x = xr.DataArray(values, coords=coords, dims=["vars", "time", "entities"])
    coords_y = {"vars": ["y"], "time": simple.times, "entities": simple.entities}
    y = xr.DataArray(
        simple.y.to_numpy().reshape(1, NENTITY, NTIME).transpose(0, 2, 1),
        coords=coords_y,
        dims=["vars", "time", "entities"],
    )
    weights = xr.DataArray(
        simple.w_grid.to_numpy()[None],
        coords={"vars": ["w"], "time": simple.times, "entities": simple.entities},
        dims=["vars", "time", "entities"],
    )
    expected = PooledOLS(y, x, weights=weights).fit().params
    flat = simple.expected_weights("grid")
    flat_params = PooledOLS(simple.y, simple.x, weights=flat).fit().params
    assert_allclose(expected, flat_params)

    with pytest.raises(ValueError, match="row index of exog"):
        PooledOLS(y, x.isel(time=slice(None, None, -1)))
    with pytest.raises(ValueError, match="row index of exog"):
        PooledOLS(y, x.isel(entities=slice(None, None, -1)))
    with pytest.raises(ValueError, match="index of weights does not match"):
        PooledOLS(y, x, weights=weights.isel(time=slice(None, None, -1)))
    with pytest.raises(ValueError, match="index of weights does not match"):
        PooledOLS(y, x, weights=weights.isel(entities=slice(None, None, -1)))
    # Both reversed
    both = PooledOLS(
        y.isel(time=slice(None, None, -1)), x.isel(time=slice(None, None, -1))
    )
    assert_allclose(both.fit().params, PooledOLS(y, x).fit().params)
    # One dimensional weights
    per_time = xr.DataArray(
        simple.w_time.to_numpy(), coords={"time": simple.times}, dims=["time"]
    )
    mod = PooledOLS(y, x, weights=per_time)
    assert_allclose(
        mod.weights.values2d[:, 0],
        simple.expected_weights("time") / simple.expected_weights("time").mean(),
    )
    with pytest.raises(ValueError, match="time index of weights"):
        PooledOLS(y, x, weights=per_time.isel(time=slice(None, None, -1)))


def test_wide_dependent_and_exog():
    # DataFrames without a MultiIndex are time (rows) by entity (columns)
    simple = Panel("str-date", "entity-major")
    y = simple.y.unstack(0)
    x = simple.x["x0"].unstack(0)
    base = PooledOLS(y, x).fit().params
    assert_allclose(base, PooledOLS(simple.y, simple.x[["x0"]]).fit().params)
    with pytest.raises(ValueError, match="row index of exog"):
        PooledOLS(y, x.iloc[::-1])
    with pytest.raises(ValueError, match="row index of exog"):
        PooledOLS(y, x[x.columns[::-1]])


def test_single_entity_wide_weights():
    # A single entity makes the time by entity grid as long as the panel
    times = pd.date_range("2001-12-31", periods=10, freq=ANNUAL_FREQ)
    index = pd.MultiIndex.from_product([["a"], times])
    rng = np.random.default_rng(3)
    y = pd.Series(rng.standard_normal(10), index=index)
    x = pd.DataFrame(rng.standard_normal((10, 1)), index=index)
    w = rng.chisquare(5, 10) + 0.5
    wide = pd.DataFrame(w[:, None], index=times, columns=["a"])
    mod = PooledOLS(y, x, weights=wide)
    assert_allclose(mod.weights.values2d[:, 0], w / w.mean())
    with pytest.raises(ValueError, match="index of weights does not match"):
        PooledOLS(y, x, weights=wide.iloc[::-1])
