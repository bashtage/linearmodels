"""
Missing values (NaN) in the entity or time index of a panel

pandas encodes a missing label in a level of an index as -1 in the codes of the
level. Operations that group by the index, such as demeaning by entity, silently
drop the rows with a missing label, and the data that they return contain NaN. An
estimator that uses them fails later with an error from the linear algebra that
does not say what is wrong, e.g., "SVD did not converge".

Every estimator has to raise the ValueError that says that the entity or the time
index contains missing values. PanelData.entity_ids and PanelData.time_ids do, but
the other operations that use the index did not, and so PanelOLS with entity
effects, PanelOLS with time effects and RandomEffects did not.
"""

from linearmodels.compat.pandas import ANNUAL_FREQ

import numpy as np
import pandas as pd
import pytest

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)

NENTITY = 20
NTIME = 5
MESSAGES = {
    "entity": "The entity index contains missing values",
    "time": "The time index contains missing values",
}


def make_panel(level, labels):
    """
    Panel where the label of the entity or the time of one row is missing

    The labels are integers, or strings for the entities and dates for the time.
    The panel has no missing labels if level is None.
    """
    rng = np.random.default_rng(0)
    if labels == "int":
        entities = np.repeat(np.arange(NENTITY, dtype=float), NTIME)
        times = np.tile(np.arange(NTIME, dtype=float), NENTITY)
        missing = np.nan
    else:
        names = np.array([f"firm{i}" for i in range(NENTITY)], dtype=object)
        entities = np.repeat(names, NTIME)
        dates = pd.date_range("2001-12-31", periods=NTIME, freq=ANNUAL_FREQ)
        times = np.tile(dates.to_numpy(), NENTITY)
        missing = None if level == "entity" else np.datetime64("NaT", "ns")
    labels_by_level = {"entity": entities, "time": times}
    if level is not None:
        labels_by_level[level][5] = missing
    index = pd.MultiIndex.from_arrays(
        [labels_by_level["entity"], labels_by_level["time"]], names=["firm", "year"]
    )
    nobs = NENTITY * NTIME
    data = pd.DataFrame(
        {
            "y": rng.standard_normal(nobs),
            "x": rng.standard_normal(nobs),
            "const": 1.0,
            "w": rng.chisquare(5, nobs) + 0.5,
            "group": np.arange(nobs) % 4,
        },
        index=index,
    )
    ncodes = {
        name: int(np.sum(data.index.codes[i] == -1))
        for i, name in enumerate(["entity", "time"])
    }
    assert ncodes == {"entity": int(level == "entity"), "time": int(level == "time")}
    return data


MODELS = {
    "pooled": lambda d: PooledOLS(d.y, d[["const", "x"]]),
    "pooled-weights": lambda d: PooledOLS(d.y, d[["const", "x"]], weights=d.w),
    "between": lambda d: BetweenOLS(d.y, d[["const", "x"]]),
    "fama-macbeth": lambda d: FamaMacBeth(d.y, d[["const", "x"]]),
    "random-effects": lambda d: RandomEffects(d.y, d[["const", "x"]]),
    "random-effects-weights": lambda d: RandomEffects(
        d.y, d[["const", "x"]], weights=d.w
    ),
    "first-difference": lambda d: FirstDifferenceOLS(d.y, d[["x"]]),
    "panel-no-effects": lambda d: PanelOLS(d.y, d[["const", "x"]]),
    "panel-entity": lambda d: PanelOLS(d.y, d[["x"]], entity_effects=True),
    "panel-entity-weights": lambda d: PanelOLS(
        d.y, d[["x"]], entity_effects=True, weights=d.w
    ),
    "panel-time": lambda d: PanelOLS(d.y, d[["x"]], time_effects=True),
    "panel-time-weights": lambda d: PanelOLS(
        d.y, d[["x"]], time_effects=True, weights=d.w
    ),
    "panel-both": lambda d: PanelOLS(
        d.y, d[["x"]], entity_effects=True, time_effects=True
    ),
    "panel-other": lambda d: PanelOLS(d.y, d[["x"]], other_effects=d[["group"]]),
    "panel-entity-other": lambda d: PanelOLS(
        d.y, d[["x"]], entity_effects=True, other_effects=d[["group"]]
    ),
}
FIT_OPTIONS = {
    "default": {},
    "robust": {"cov_type": "robust"},
    "kernel": {"cov_type": "kernel"},
    "clustered-entity": {"cov_type": "clustered", "cluster_entity": True},
    "clustered-time": {"cov_type": "clustered", "cluster_time": True},
}


@pytest.mark.parametrize("labels", ["int", "str-date"])
@pytest.mark.parametrize("level", ["entity", "time"])
@pytest.mark.parametrize("model", MODELS)
def test_every_estimator_rejects_a_missing_label(model, level, labels):
    # LinAlgError (SVD did not converge) was raised by PanelOLS with entity effects
    # and a missing entity and by PanelOLS with time effects and a missing time.
    # RandomEffects with a missing entity raised "array must not contain infs or
    # NaNs"
    data = make_panel(level, labels)
    with pytest.raises(ValueError, match=MESSAGES[level]) as exc_info:
        MODELS[model](data).fit()
    assert exc_info.type is ValueError


@pytest.mark.parametrize("level", ["entity", "time"])
@pytest.mark.parametrize("options", FIT_OPTIONS)
@pytest.mark.parametrize("model", ["panel-entity", "panel-both", "random-effects"])
def test_covariance_estimators(model, options, level):
    data = make_panel(level, "int")
    with pytest.raises(ValueError, match=MESSAGES[level]):
        MODELS[model](data).fit(**FIT_OPTIONS[options])


@pytest.mark.parametrize("level", ["entity", "time"])
def test_low_memory_demeaning_by_both(level):
    data = make_panel(level, "int")
    mod = PanelOLS(data.y, data[["x"]], entity_effects=True, time_effects=True)
    with pytest.raises(ValueError, match=MESSAGES[level]):
        mod.fit(low_memory=True)


@pytest.mark.parametrize("level", ["entity", "time"])
@pytest.mark.parametrize(
    ("model", "formula"),
    [
        (PanelOLS, "y ~ x + EntityEffects + TimeEffects"),
        (PanelOLS, "y ~ x + EntityEffects"),
        (RandomEffects, "y ~ 1 + x"),
        (PooledOLS, "y ~ 1 + x"),
    ],
)
def test_formulas(model, formula, level):
    data = make_panel(level, "int")
    with pytest.raises(ValueError, match=MESSAGES[level]):
        model.from_formula(formula, data).fit()


# ----------------------------------------------------------------------------
# PanelData, whose methods the estimators use
# ----------------------------------------------------------------------------


def weights(panel):
    return PanelData(panel.dataframe[["w"]])


ENTITY = ("entity",)
TIME = ("time",)
BOTH = ("entity", "time")
# The levels of the index that each operation groups by, and the operation
OPERATIONS = {
    "entity_ids": (ENTITY, lambda p: p.entity_ids),
    "time_ids": (TIME, lambda p: p.time_ids),
    "demean-entity": (ENTITY, lambda p: p.demean("entity")),
    "demean-time": (TIME, lambda p: p.demean("time")),
    "demean-both": (BOTH, lambda p: p.demean("both")),
    "demean-both-low-memory": (BOTH, lambda p: p.demean("both", low_memory=True)),
    "demean-entity-array": (ENTITY, lambda p: p.demean("entity", return_panel=False)),
    "demean-entity-weights": (ENTITY, lambda p: p.demean("entity", weights=weights(p))),
    "demean-time-weights": (TIME, lambda p: p.demean("time", weights=weights(p))),
    "mean-entity": (ENTITY, lambda p: p.mean("entity")),
    "mean-time": (TIME, lambda p: p.mean("time")),
    "mean-entity-weights": (ENTITY, lambda p: p.mean("entity", weights=weights(p))),
    "count-entity": (ENTITY, lambda p: p.count("entity")),
    "count-time": (TIME, lambda p: p.count("time")),
    "dummies-entity": (ENTITY, lambda p: p.dummies("entity")),
    "dummies-time": (TIME, lambda p: p.dummies("time")),
    "first_difference": (BOTH, lambda p: p.first_difference()),
    "values3d": (BOTH, lambda p: p.values3d),
    "panel": (BOTH, lambda p: p.panel),
}
CASES = [
    pytest.param(name, level, id=f"{name}-{level}")
    for name, (levels, _) in OPERATIONS.items()
    for level in levels
]


@pytest.mark.parametrize(("operation", "level"), CASES)
def test_panel_data_operations(operation, level):
    # The rows with a missing label are not in an entity or time period, so the
    # results have no meaning. Most of these returned data that contain NaN or that
    # use the last label for the missing one
    data = make_panel(level, "int")
    panel = PanelData(data[["y", "x", "w"]])
    with pytest.raises(ValueError, match=MESSAGES[level]) as exc_info:
        OPERATIONS[operation][1](panel)
    assert exc_info.type is ValueError


@pytest.mark.parametrize("operation", OPERATIONS)
def test_panel_data_operations_without_a_missing_label(operation):
    panel = PanelData(make_panel(None, "int")[["y", "x", "w"]])
    assert OPERATIONS[operation][1](panel) is not None


@pytest.mark.parametrize("model", MODELS)
def test_estimators_without_a_missing_label(model):
    res = MODELS[model](make_panel(None, "int")).fit()
    assert np.all(np.isfinite(res.params))
