"""
The log-likelihood of weighted panel models

The log-likelihood of a model with weights w is

    -n/2 (ln(2 pi) + ln(s2) + 1) + 1/2 sum(ln(w)),    s2 = sum(w e^2) / n,

the Gaussian log-likelihood of the regression of the original data where the variance
of the error of an observation is sigma^2 / w. The last term is the Jacobian of the
weights. It was omitted, so that the log-likelihood was that of the regression of the
data multiplied by sqrt(w). That is not the log-likelihood of the data, and it is not
comparable between different weights. Models without weights do not change.

The references are the log-likelihood of the weighted least squares regression of
statsmodels (WLS(...).llf), of the regression that each model estimates, written out
here, and the formula.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.api as sm

from linearmodels.datasets import wage_panel
from linearmodels.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)

NENTITY = 30
NPERIOD = 8


def make_panel(unbalanced, seed=3, staggered=False):
    """A panel with heteroskedastic errors, a variance proportional to 1 / w"""
    rg = np.random.default_rng(seed)
    rows = []
    for entity in range(NENTITY):
        if staggered:
            # Consecutive periods only, so that no entity has a gap
            first = rg.integers(0, 3)
            last = NPERIOD - 1 - rg.integers(0, 3)
            periods = np.arange(first, last + 1)
        elif unbalanced:
            keep = rg.random(NPERIOD) > 0.25
            keep[rg.choice(NPERIOD, 3, replace=False)] = True
            periods = np.flatnonzero(keep)
        else:
            periods = np.arange(NPERIOD)
        rows.extend((entity, int(t)) for t in periods)
    index = pd.MultiIndex.from_tuples(rows, names=["entity", "time"])
    nobs = index.shape[0]
    entity_effect = rg.standard_normal(NENTITY)[index.codes[0]]
    x1 = rg.standard_normal(nobs) + 0.4 * entity_effect
    x2 = rg.standard_normal(nobs)
    weight = rg.chisquare(4, nobs) / 4 + 0.2
    shock = rg.standard_normal(nobs) / np.sqrt(weight)
    y = 1 + 0.5 * x1 - 0.3 * x2 + entity_effect + shock
    group = (index.codes[0] + index.get_level_values(1).to_numpy()) % 4
    return pd.DataFrame(
        {"y": y, "const": 1.0, "x1": x1, "x2": x2, "w": weight, "g": group},
        index=index,
    )


def gaussian_loglik(wresid, weights):
    """The formula of the log-likelihood"""
    nobs = wresid.shape[0]
    sigma2 = float(wresid @ wresid) / nobs
    jacobian = 0.5 * np.sum(np.log(weights))
    return -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1) + jacobian


@pytest.fixture(scope="module", params=[False, True], ids=["balanced", "unbalanced"])
def frame(request):
    return make_panel(request.param)


# --------------------------------------------------------------------------
# PooledOLS
# --------------------------------------------------------------------------
def test_pooled_ols_against_statsmodels(frame):
    res = PooledOLS(frame["y"], frame[["const", "x1", "x2"]], weights=frame["w"]).fit()
    wls = sm.WLS(frame["y"], frame[["const", "x1", "x2"]], weights=frame["w"]).fit()
    assert_allclose(res.loglik, wls.llf, rtol=1e-10)


def test_pooled_ols_wage_panel_against_statsmodels():
    data = wage_panel.load().set_index(["nr", "year"])
    data["const"] = 1.0
    exog = data[["const", "educ", "married", "union"]]
    res = PooledOLS(data["lwage"], exog, weights=data["hours"]).fit()
    wls = sm.WLS(data["lwage"], exog, weights=data["hours"]).fit()
    assert_allclose(res.loglik, wls.llf, rtol=1e-10)


def test_pooled_ols_formula(frame):
    res = PooledOLS(frame["y"], frame[["const", "x1", "x2"]], weights=frame["w"]).fit()
    # The weights have a mean of one
    weights = frame["w"].to_numpy() / frame["w"].mean()
    wresid = np.asarray(res.wresids)
    assert_allclose(res.loglik, gaussian_loglik(wresid, weights), rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-6, 0.1, 37.0, 1e8])
def test_pooled_ols_scale_of_the_weights_does_not_matter(frame, scale):
    exog = frame[["const", "x1", "x2"]]
    expected = PooledOLS(frame["y"], exog, weights=frame["w"]).fit().loglik
    res = PooledOLS(frame["y"], exog, weights=scale * frame["w"]).fit()
    assert_allclose(res.loglik, expected, rtol=1e-10)


def test_pooled_ols_unweighted_is_ols(frame):
    res = PooledOLS(frame["y"], frame[["const", "x1", "x2"]]).fit()
    ols = sm.OLS(frame["y"], frame[["const", "x1", "x2"]]).fit()
    assert_allclose(res.loglik, ols.llf, rtol=1e-12)


@pytest.mark.parametrize("model", ["PooledOLS", "PanelOLS", "RandomEffects"])
def test_equal_weights_are_no_weights(frame, model):
    def fit(weights):
        if model == "PooledOLS":
            mod = PooledOLS(frame["y"], frame[["const", "x1", "x2"]], weights=weights)
        elif model == "PanelOLS":
            mod = PanelOLS(
                frame["y"], frame[["x1", "x2"]], weights=weights, entity_effects=True
            )
        else:
            mod = RandomEffects(
                frame["y"], frame[["const", "x1", "x2"]], weights=weights
            )
        return mod.fit()

    weights = pd.Series(5.0, index=frame.index)
    assert_allclose(fit(weights).loglik, fit(None).loglik, rtol=1e-12)


# --------------------------------------------------------------------------
# PanelOLS: the regression on the dummy variables of the effects
# --------------------------------------------------------------------------
EFFECTS = {
    "entity": {"entity_effects": True},
    "time": {"time_effects": True},
    "both": {"entity_effects": True, "time_effects": True},
    "other": {"other_effects": ["g"]},
    "entity+other": {"entity_effects": True, "other_effects": ["g"]},
}
ALGORITHMS = {"means": {}, "lsdv": {"use_lsdv": True}, "lsmr": {"use_lsmr": True}}


def dummies(values, prefix, index, drop_first):
    """The dummy variables of the groups"""
    group = pd.Series(np.asarray(values), index=index)
    out = pd.get_dummies(group, dtype=float, drop_first=drop_first)
    out.columns = [f"{prefix}.{c}" for c in out.columns]
    return out


def lsdv_design(frame, exog_columns, effect):
    """
    The regressors and the dummy variables of the groups of each effect

    A dummy variable is dropped from the effects after the first, and from the first
    if there is a constant, so that the design has full column rank. A pseudo inverse
    of exactly collinear dummy variables does not give the least squares solution to
    the accuracy that is needed.
    """
    design = [frame[exog_columns]]
    drop_first = "const" in exog_columns
    if effect in ("entity", "both", "entity+other"):
        values = frame.index.get_level_values(0)
        design.append(dummies(values, "entity", frame.index, drop_first))
        drop_first = True
    if effect in ("time", "both"):
        values = frame.index.get_level_values(1)
        design.append(dummies(values, "time", frame.index, drop_first))
        drop_first = True
    if effect in ("other", "entity+other"):
        design.append(dummies(frame["g"], "group", frame.index, drop_first))
    return pd.concat(design, axis=1)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("effect", EFFECTS)
@pytest.mark.parametrize("constant", [False, True], ids=["no-constant", "constant"])
def test_panel_ols_against_dummy_variable_regression(
    frame, constant, effect, algorithm
):
    columns = ["const", "x1", "x2"] if constant else ["x1", "x2"]
    options = dict(EFFECTS[effect])
    if "other_effects" in options:
        options["other_effects"] = frame[options["other_effects"]]
    mod = PanelOLS(frame["y"], frame[columns], weights=frame["w"], **options)
    res = mod.fit(**ALGORITHMS[algorithm])
    design = lsdv_design(frame, columns, effect)
    wls = sm.WLS(frame["y"], design, weights=frame["w"]).fit()
    assert_allclose(res.loglik, wls.llf, rtol=1e-8)


@pytest.mark.parametrize("effect", EFFECTS)
def test_panel_ols_unweighted_against_dummy_variable_regression(frame, effect):
    options = dict(EFFECTS[effect])
    if "other_effects" in options:
        options["other_effects"] = frame[options["other_effects"]]
    res = PanelOLS(frame["y"], frame[["x1", "x2"]], **options).fit()
    ols = sm.OLS(frame["y"], lsdv_design(frame, ["x1", "x2"], effect)).fit()
    assert_allclose(res.loglik, ols.llf, rtol=1e-8)


# --------------------------------------------------------------------------
# BetweenOLS: the regression of the means of the entities
# --------------------------------------------------------------------------
def between_regression(frame, weighted, reweight):
    grouped = frame.groupby(level=0)
    w = frame["w"] if weighted else pd.Series(1.0, index=frame.index)
    sum_w = w.groupby(level=0).sum()
    columns = ["y", "const", "x1", "x2"]
    means = (frame[columns].mul(w, axis=0)).groupby(level=0).sum().div(sum_w, axis=0)
    if weighted:
        weights = sum_w
    elif reweight:
        weights = grouped.size().astype(float)
    else:
        weights = pd.Series(1.0, index=sum_w.index)
    return means, weights


@pytest.mark.parametrize(
    ("weighted", "reweight"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["unweighted", "reweight", "weighted", "weighted-reweight"],
)
def test_between_ols_against_statsmodels(frame, weighted, reweight):
    w = frame["w"] if weighted else None
    mod = BetweenOLS(frame["y"], frame[["const", "x1", "x2"]], weights=w)
    res = mod.fit(reweight=reweight)
    means, weights = between_regression(frame, weighted, reweight)
    wls = sm.WLS(means["y"], means[["const", "x1", "x2"]], weights=weights).fit()
    assert_allclose(res.loglik, wls.llf, rtol=1e-10)
    # The regression has the entities as observations
    assert res.nobs == NENTITY


# --------------------------------------------------------------------------
# FirstDifferenceOLS: the regression of the first differences
# --------------------------------------------------------------------------
@pytest.mark.parametrize("staggered", [False, True], ids=["balanced", "staggered"])
def test_first_difference_ols_against_statsmodels(staggered):
    frame = make_panel(False, staggered=staggered)
    mod = FirstDifferenceOLS(frame["y"], frame[["x1", "x2"]], weights=frame["w"])
    res = mod.fit()
    by_entity = frame.groupby(level=0)
    diffs = by_entity[["y", "x1", "x2"]].diff().dropna()
    previous = by_entity["w"].shift(1).loc[diffs.index]
    weights = 1.0 / (1.0 / frame["w"].loc[diffs.index] + 1.0 / previous)
    wls = sm.WLS(diffs["y"], diffs[["x1", "x2"]], weights=weights).fit()
    assert res.nobs == diffs.shape[0]
    assert_allclose(res.loglik, wls.llf, rtol=1e-10)


def test_first_difference_ols_unweighted():
    frame = make_panel(False)
    res = FirstDifferenceOLS(frame["y"], frame[["x1", "x2"]]).fit()
    diffs = frame.groupby(level=0)[["y", "x1", "x2"]].diff().dropna()
    ols = sm.OLS(diffs["y"], diffs[["x1", "x2"]]).fit()
    assert_allclose(res.loglik, ols.llf, rtol=1e-12)


# --------------------------------------------------------------------------
# RandomEffects: the regression of the quasi-demeaned data
# --------------------------------------------------------------------------
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_random_effects_against_statsmodels(frame, weighted):
    w = frame["w"] if weighted else None
    res = RandomEffects(frame["y"], frame[["const", "x1", "x2"]], weights=w).fit()
    weights = frame["w"] if weighted else pd.Series(1.0, index=frame.index)
    columns = ["y", "const", "x1", "x2"]
    sum_w = weights.groupby(level=0).sum()
    means = frame[columns].mul(weights, axis=0).groupby(level=0).sum()
    means = means.div(sum_w, axis=0)
    # The weighted mean of the entity and theta of the entity of each observation
    entity = frame.index.get_level_values(0)
    mean_rows = means.loc[entity].to_numpy()
    theta_rows = res.theta["theta"].loc[entity].to_numpy()[:, None]
    transformed = pd.DataFrame(
        frame[columns].to_numpy() - theta_rows * mean_rows,
        index=frame.index,
        columns=columns,
    )
    wls = sm.WLS(
        transformed["y"], transformed[["const", "x1", "x2"]], weights=weights
    ).fit()
    assert_allclose(res.loglik, wls.llf, rtol=1e-8)


# --------------------------------------------------------------------------
# FamaMacBeth: the pooled residuals at the average of the estimates
# --------------------------------------------------------------------------
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_fama_macbeth_formula(frame, weighted):
    w = frame["w"] if weighted else None
    res = FamaMacBeth(frame["y"], frame[["const", "x1", "x2"]], weights=w).fit()
    weights = frame["w"] / frame["w"].mean() if weighted else np.ones(frame.shape[0])
    resid = frame["y"] - frame[["const", "x1", "x2"]] @ res.params
    wresid = np.sqrt(np.asarray(weights)) * np.asarray(resid)
    assert_allclose(
        res.loglik, gaussian_loglik(wresid, np.asarray(weights)), rtol=1e-10
    )


def test_loglik_is_the_loglik_of_the_weighted_data_plus_the_jacobian(frame):
    exog = frame[["const", "x1", "x2"]]
    weighted = PooledOLS(frame["y"], exog, weights=frame["w"]).fit()
    wls = sm.WLS(frame["y"], exog, weights=frame["w"] / frame["w"].mean()).fit()
    jacobian = 0.5 * np.sum(np.log(frame["w"] / frame["w"].mean()))
    n = weighted.nobs
    without = -0.5 * n * (np.log(2 * np.pi) + np.log(weighted.resid_ss / n) + 1)
    assert_allclose(weighted.loglik, without + jacobian, rtol=1e-12)
    assert_allclose(weighted.loglik, wls.llf, rtol=1e-10)
