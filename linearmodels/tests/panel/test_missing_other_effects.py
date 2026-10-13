"""
Missing values in other_effects of PanelOLS

An observation where one of the other effects is missing is dropped with the
observations that have a missing value in dependent, exog or weights, and a
MissingValueWarning is issued. It was given a category of its own, and fit failed
after the estimation with an error about the observations per group.
"""

import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.api as sm

from linearmodels.panel import PanelOLS
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.tests.panel._data_checks import (
    PERIODS,
    assert_equal_results,
    make_frame,
)

MISSING_ROWS = [4, 17, 18, 40, 41, 63]
FIT_ALGORITHMS = [
    pytest.param({}, id="means"),
    pytest.param({"use_lsdv": True}, id="lsdv"),
    pytest.param({"use_lsmr": True}, id="lsmr"),
]


@pytest.fixture(scope="module")
def frame():
    return make_frame(PERIODS.to_timestamp())


def effects_with_missing(frame, rows, kind="float"):
    """The category g with missing values in the observations ``rows``"""
    if kind == "float":
        g = frame["g"].astype(float)
        g.iloc[rows] = np.nan
    elif kind == "object":
        g = frame["g"].map(lambda v: f"cat{v}").astype(object)
        g.iloc[rows] = None
    else:
        g = pd.Series(
            pd.Categorical(frame["g"].map(lambda v: f"cat{v}")), index=frame.index
        )
        g.iloc[rows] = np.nan
    return g.to_frame("g")


def compare_with_manual_drop(frame, kind, weights, algorithm, entity, singletons=True):
    effects = effects_with_missing(frame, MISSING_ROWS, kind)
    missing = effects["g"].isna().to_numpy()
    clean = frame.loc[~missing]
    w = frame["w"] if weights else None
    w_clean = clean["w"] if weights else None
    options = {"entity_effects": entity, "singletons": singletons}
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(
            frame["y"], frame[["x1", "x2"]], weights=w, other_effects=effects, **options
        )
    manual = PanelOLS(
        clean["y"],
        clean[["x1", "x2"]],
        weights=w_clean,
        other_effects=effects.loc[~missing],
        **options,
    )
    assert mod.not_null.sum() == (~missing).sum()
    cov_options = {"cov_type": "clustered", "cluster_entity": True}
    res = mod.fit(**algorithm, **cov_options)
    expected = manual.fit(**algorithm, **cov_options)
    assert res.nobs == (~missing).sum()
    assert_equal_results(res, expected)
    return res, expected


@pytest.mark.parametrize("entity", [False, True], ids=["other", "entity+other"])
@pytest.mark.parametrize("weights", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("algorithm", FIT_ALGORITHMS)
@pytest.mark.parametrize("kind", ["float", "object", "categorical"])
def test_missing_other_effects_are_dropped(frame, kind, algorithm, weights, entity):
    compare_with_manual_drop(frame, kind, weights, algorithm, entity)


def test_missing_other_effects_with_two_other_effects(frame):
    effects = effects_with_missing(frame, MISSING_ROWS)
    effects["h"] = (np.arange(frame.shape[0]) // 3) % 5
    effects.iloc[[2, 30], 1] = np.nan
    missing = effects.isna().any(axis=1).to_numpy()
    assert missing.sum() == len(MISSING_ROWS) + 2
    clean = frame.loc[~missing]
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(
            frame["y"],
            frame[["x1", "x2"]],
            weights=frame["w"],
            other_effects=effects,
        )
    manual = PanelOLS(
        clean["y"],
        clean[["x1", "x2"]],
        weights=clean["w"],
        other_effects=effects.loc[~missing],
    )
    assert_equal_results(mod.fit(), manual.fit())


def test_missing_other_effects_with_singletons_dropped(frame):
    res, _ = compare_with_manual_drop(
        frame, "float", True, {}, entity=False, singletons=False
    )
    assert res.nobs <= (frame.shape[0] - len(MISSING_ROWS))


@pytest.mark.parametrize("weights", [False, True], ids=["unweighted", "weighted"])
def test_missing_other_effects_against_dummy_variables(frame, weights):
    # The effects of the observations that remain, estimated by least squares on
    # the dummy variables of the categories and of the entities
    effects = effects_with_missing(frame, MISSING_ROWS)
    keep = effects["g"].notna().to_numpy()
    clean = frame.loc[keep]
    w = frame["w"] if weights else None
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(
            frame["y"],
            frame[["x1", "x2"]],
            weights=w,
            entity_effects=True,
            other_effects=effects,
        )
    res = mod.fit()
    dummies = pd.get_dummies(effects.loc[keep, "g"], dtype=float, drop_first=True)
    entity = pd.get_dummies(
        clean.index.get_level_values(0), dtype=float, drop_first=False
    ).set_axis(clean.index)
    design = pd.concat([clean[["x1", "x2"]], entity, dummies], axis=1)
    if weights:
        # The weights of the model have a mean of one, which sets the scale of the
        # sums of squares
        w_clean = clean["w"] / clean["w"].mean()
        ols = sm.WLS(clean["y"], design, weights=w_clean).fit()
    else:
        ols = sm.OLS(clean["y"], design).fit()
    assert_allclose(res.params, ols.params[["x1", "x2"]])
    assert res.nobs == ols.nobs
    assert_allclose(res.resid_ss, ols.ssr)
    assert res.df_resid == ols.df_resid


def test_missing_other_effects_are_not_double_counted(frame):
    # An observation that is also missing in y and x is dropped once
    data = frame.copy()
    data.iloc[MISSING_ROWS[:2], data.columns.get_loc("y")] = np.nan
    data.iloc[MISSING_ROWS[1:3], data.columns.get_loc("x1")] = np.nan
    effects = effects_with_missing(frame, MISSING_ROWS)
    missing = (effects["g"].isna() | data[["y", "x1"]].isna().any(axis=1)).to_numpy()
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(data["y"], data[["x1", "x2"]], other_effects=effects)
    assert mod.not_null.sum() == (~missing).sum()
    manual = PanelOLS(
        data["y"][~missing],
        data.loc[~missing, ["x1", "x2"]],
        other_effects=effects.loc[~missing],
    )
    assert_equal_results(mod.fit(), manual.fit())


def test_missing_other_effects_clusters_and_predict(frame):
    effects = effects_with_missing(frame, MISSING_ROWS)
    missing = effects["g"].isna().to_numpy()
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(frame["y"], frame[["x1", "x2"]], other_effects=effects)
    # The clusters have a value for each row of the data that were given
    res = mod.fit(cov_type="clustered", clusters=frame[["g"]])
    manual = PanelOLS(
        frame["y"][~missing],
        frame.loc[~missing, ["x1", "x2"]],
        other_effects=effects.loc[~missing],
    ).fit(cov_type="clustered", clusters=frame.loc[~missing, ["g"]])
    assert_equal_results(res, manual)
    # The original index is that of the data before the observations were dropped
    out = res.predict(effects=True, idiosyncratic=True, missing=True)
    assert out.index.equals(frame.index)
    assert out.iloc[MISSING_ROWS].isna().all().all()
    assert out.drop(frame.index[MISSING_ROWS]).notna().all().all()


def test_no_warning_without_missing_other_effects(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mod = PanelOLS(frame["y"], frame[["x1", "x2"]], other_effects=frame[["g"]])
    assert mod.not_null.all()


def test_missing_other_effects_numpy_array(frame):
    # A 1-d array has a value for each row of dependent
    values = frame["g"].to_numpy().astype(float)
    values[MISSING_ROWS] = np.nan
    missing = np.isnan(values)
    with pytest.warns(MissingValueWarning):
        mod = PanelOLS(frame["y"], frame[["x1", "x2"]], other_effects=values)
    manual = PanelOLS(
        frame["y"][~missing],
        frame.loc[~missing, ["x1", "x2"]],
        other_effects=values[~missing],
    )
    assert_equal_results(mod.fit(), manual.fit())


def test_other_effects_with_the_wrong_number_of_rows(frame):
    effects = frame[["g"]].iloc[:-3]
    with pytest.raises(ValueError, match=r"other_effects has \d+ rows, but dependent"):
        PanelOLS(frame["y"], frame[["x1", "x2"]], other_effects=effects)
