"""
PanelOLS with singletons=False and the clusters of the entities or the time periods

The singleton observations are dropped from the data of the model, and cluster_entity and
cluster_time (which are made from the data of the model) have to have the observations
that remain. They raised "IndexError: Boolean index has wrong length", since the mask of
the singletons, which has a value for each observation before they were dropped, was also
applied to the clusters that were made after. The results have to be those of the model
estimated from the data without the singleton observations.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.panel import PanelOLS
from linearmodels.shared.exceptions import MissingValueWarning, SingletonWarning

NENTITY = 30
NPERIOD = 6


def make_data(seed=0, nsingleton=3):
    """A panel in which the first entities have a single observation"""
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(NENTITY), np.arange(NPERIOD)], names=["entity", "time"]
    )
    entity = index.codes[0]
    x = rg.standard_normal((index.shape[0], 2))
    effect = rg.standard_normal(NENTITY)[entity]
    shock = (1 + np.abs(x[:, 0])) * rg.standard_normal(index.shape[0])
    data = pd.DataFrame(
        {
            "y": x @ [0.5, -0.3] + effect + shock,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "w": rg.chisquare(4, index.shape[0]) / 4 + 0.2,
            "group": entity // 5,
        },
        index=index,
    )
    # Entities 3, 7, ... keep a single observation
    keep = np.ones(index.shape[0], dtype=bool)
    for i in range(nsingleton):
        rows = np.flatnonzero(entity == 3 + 4 * i)
        keep[rows[1:]] = False
    return data.loc[keep]


def drop_singletons(data, effects):
    """Drop the observations that are alone in a group of an effect, repeatedly"""
    while True:
        alone = np.zeros(data.shape[0], dtype=bool)
        for level in effects:
            keys = data.index.get_level_values(level)
            alone |= data.groupby(keys)["y"].transform("size").to_numpy() == 1
        if not alone.any():
            return data
        data = data.loc[~alone]


COVARIANCES = {
    "entity": {"cluster_entity": True},
    "time": {"cluster_time": True},
    "entity-time": {"cluster_entity": True, "cluster_time": True},
}
EFFECTS = {
    "entity": ({"entity_effects": True}, ["entity"]),
    "both": ({"entity_effects": True, "time_effects": True}, ["entity", "time"]),
}


def compare(res, expected):
    assert res.nobs == expected.nobs
    assert_allclose(res.params, expected.params)
    assert_allclose(res.cov, expected.cov)
    assert_allclose(res.std_errors, expected.std_errors)
    assert_allclose(res.f_statistic_robust.stat, expected.f_statistic_robust.stat)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("clustering", list(COVARIANCES))
@pytest.mark.parametrize("effect", list(EFFECTS))
def test_singleton_observations_with_clusters_of_the_model(
    effect, clustering, weighted
):
    data = make_data()
    options, levels = EFFECTS[effect]
    weights = data["w"] if weighted else None
    with pytest.warns(SingletonWarning):
        mod = PanelOLS(
            data["y"], data[["x1", "x2"]], weights=weights, singletons=False, **options
        )
    res = mod.fit(cov_type="clustered", **COVARIANCES[clustering])
    kept = drop_singletons(data, levels)
    assert kept.shape[0] < data.shape[0]
    manual_weights = kept["w"] if weighted else None
    expected = PanelOLS(
        kept["y"], kept[["x1", "x2"]], weights=manual_weights, **options
    ).fit(cov_type="clustered", **COVARIANCES[clustering])
    compare(res, expected)


@pytest.mark.parametrize("clustering", ["entity", "time"])
def test_singleton_observations_with_clusters_and_the_clusters_of_the_model(clustering):
    # The clusters that are given have a value for each observation of the data, and
    # are two-way clusters together with the cluster of the entities or of the periods
    data = make_data()
    with pytest.warns(SingletonWarning):
        mod = PanelOLS(
            data["y"], data[["x1", "x2"]], singletons=False, entity_effects=True
        )
    res = mod.fit(
        cov_type="clustered", clusters=data[["group"]], **COVARIANCES[clustering]
    )
    kept = drop_singletons(data, ["entity"])
    expected = PanelOLS(kept["y"], kept[["x1", "x2"]], entity_effects=True).fit(
        cov_type="clustered", clusters=kept[["group"]], **COVARIANCES[clustering]
    )
    compare(res, expected)


@pytest.mark.parametrize("clustering", ["entity", "time"])
def test_clusters_of_the_model_are_not_set_on_a_slice_of_the_clusters(clustering):
    # The clusters without the singletons were a slice of the clusters that are given,
    # and the clusters of the entities or periods were set on it. pandas 2 warns "A
    # value is trying to be set on a copy of a slice from a DataFrame" (the warning
    # does not exist in pandas 3, where the test passes whatever the code does)
    data = make_data()
    with pytest.warns(SingletonWarning):
        mod = PanelOLS(
            data["y"], data[["x1", "x2"]], singletons=False, entity_effects=True
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = mod.fit(
            cov_type="clustered", clusters=data[["group"]], **COVARIANCES[clustering]
        )
    categories = [w.category.__name__ for w in caught]
    assert "SettingWithCopyWarning" not in categories
    # Nothing else changes: the clusters are those of the model without singletons
    kept = drop_singletons(data, ["entity"])
    expected = PanelOLS(kept["y"], kept[["x1", "x2"]], entity_effects=True).fit(
        cov_type="clustered", clusters=kept[["group"]], **COVARIANCES[clustering]
    )
    compare(res, expected)


@pytest.mark.parametrize("clustering", ["entity", "time"])
def test_singleton_observations_with_a_grid_of_clusters(clustering):
    # The panel is not balanced since the singletons only have one observation. The
    # clusters for each time period and entity are those of the clusters for each row
    data = make_data()
    grid = data["group"].unstack(0)
    assert grid.shape == (NPERIOD, NENTITY)
    with pytest.warns(SingletonWarning):
        mod = PanelOLS(
            data["y"], data[["x1", "x2"]], singletons=False, entity_effects=True
        )
    res = mod.fit(cov_type="clustered", clusters=grid, **COVARIANCES[clustering])
    expected = mod.fit(
        cov_type="clustered", clusters=data[["group"]], **COVARIANCES[clustering]
    )
    compare(res, expected)
    kept = drop_singletons(data, ["entity"])
    manual = PanelOLS(kept["y"], kept[["x1", "x2"]], entity_effects=True).fit(
        cov_type="clustered", clusters=kept[["group"]], **COVARIANCES[clustering]
    )
    compare(res, manual)


def test_singleton_observations_with_missing_values_and_clusters():
    # The missing observations are dropped before the singletons, and the singletons
    # are the observations that remain alone
    data = make_data()
    missing = data.copy()
    missing.iloc[[5, 6, 40, 41, 42, 90], missing.columns.get_loc("x1")] = np.nan
    with pytest.warns((SingletonWarning, MissingValueWarning)):
        mod = PanelOLS(
            missing["y"], missing[["x1", "x2"]], singletons=False, entity_effects=True
        )
    res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    kept = drop_singletons(missing.dropna(subset=["x1"]), ["entity"])
    expected = PanelOLS(kept["y"], kept[["x1", "x2"]], entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    compare(res, expected)
