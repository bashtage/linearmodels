"""
Statistics of panel models that use weights or that transform the data unevenly

The classical (homoskedastic) F-statistic tests that the slopes are zero in the
regression of the transformed data. It has to be the Wald statistic of the same
hypothesis that is computed from the unadjusted covariance of the parameters, and
it has to be the F-statistic of the same regression in other packages. These are
the references:

* PooledOLS and BetweenOLS: the F-statistic of a weighted least squares regression
  of statsmodels, for the observations and for the means of the entities.
* PanelOLS: the F-test that the slopes are zero in a weighted least squares
  regression on the dummy variables of the effects.
* RandomEffects: there is no other implementation of the model with these weights,
  and the F-statistic is the Wald statistic of the unadjusted covariance.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.api as sm

from linearmodels.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


def make_data(unbalanced, seed=0, nentity=24, nperiod=6):
    """A panel where the weights, the regressors and the errors differ by entity"""
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    entity = index.codes[0]
    period = index.codes[1]
    nobs = nentity * nperiod
    effect = rg.standard_normal(nentity)[entity]
    time_effect = rg.standard_normal(nperiod)[period]
    x1 = 0.5 * effect + rg.standard_normal(nobs)
    x2 = rg.standard_normal(nobs) + 0.3 * time_effect
    shock = (0.5 + np.abs(x1)) * rg.standard_normal(nobs)
    data = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * x1 - 0.5 * x2 + effect + 0.5 * time_effect + shock,
            "const": 1.0,
            "x1": x1,
            "x2": x2,
            "w": rg.chisquare(4, nobs) / 4 + 0.2,
            "group": (entity + 2 * period) % 5,
            "entity": entity,
            "time": period,
        },
        index=index,
    )
    if unbalanced:
        keep = rg.random((nentity, nperiod)) > 0.3
        for i in range(nentity):
            keep[i, rg.choice(nperiod, 3, replace=False)] = True
        data = data.loc[keep.ravel()]
    return data


def effects_f_statistic(data, weights, categories):
    """The F-statistic of the slopes in a regression on the dummies of the effects"""
    weights = np.ones(data.shape[0]) if weights is None else weights
    dummies = pd.concat(
        [
            pd.get_dummies(data[c], prefix=c, drop_first=i > 0).astype(float)
            for i, c in enumerate(categories)
        ],
        axis=1,
    )
    full = sm.WLS(
        data["y"], pd.concat([data[["x1", "x2"]], dummies], axis=1), weights=weights
    ).fit()
    restricted = sm.WLS(data["y"], dummies, weights=weights).fit()
    stat = (restricted.ssr - full.ssr) / 2 / (full.ssr / full.df_resid)
    return stat, full.df_resid


WEIGHTED = [False, True]
UNBALANCED = [False, True]
EFFECTS = {
    "entity": ({"entity_effects": True}, ["entity"]),
    "time": ({"time_effects": True}, ["time"]),
    "both": ({"entity_effects": True, "time_effects": True}, ["entity", "time"]),
    "other": ({"other_effects": ("group",)}, ["group"]),
    "entity-other": (
        {"entity_effects": True, "other_effects": ("group",)},
        ["entity", "group"],
    ),
}


@pytest.fixture(params=UNBALANCED, ids=["balanced", "unbalanced"])
def data(request):
    return make_data(request.param)


def weights_of(data, weighted):
    return data["w"] if weighted else None


@pytest.mark.parametrize("weighted", WEIGHTED)
def test_pooled_f_statistic_against_wls(data, weighted):
    weights = weights_of(data, weighted)
    res = PooledOLS(data["y"], data[["const", "x1", "x2"]], weights=weights).fit()
    ref_weights = np.ones(data.shape[0]) if weights is None else weights
    ref = sm.WLS(data["y"], data[["const", "x1", "x2"]], weights=ref_weights).fit()
    assert_allclose(res.f_statistic.stat, ref.fvalue)
    assert res.f_statistic.df == 2
    assert res.f_statistic.df_denom == ref.df_resid


@pytest.mark.parametrize("weighted", WEIGHTED)
@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("effect", list(EFFECTS))
def test_panel_f_statistic_against_dummy_regression(data, effect, constant, weighted):
    options, categories = EFFECTS[effect]
    options = dict(options)
    if "other_effects" in options:
        options["other_effects"] = data[list(options["other_effects"])]
    columns = ["const", "x1", "x2"] if constant else ["x1", "x2"]
    weights = weights_of(data, weighted)
    res = PanelOLS(data["y"], data[columns], weights=weights, **options).fit()
    stat, df_denom = effects_f_statistic(data, weights, categories)
    assert_allclose(res.f_statistic.stat, stat)
    assert res.f_statistic.df == 2
    assert res.f_statistic.df_denom == df_denom


def entity_means(data, weights):
    """The (weighted) means of the entities and the total weight of each entity"""
    columns = ["y", "const", "x1", "x2"]
    weights = np.ones(data.shape[0]) if weights is None else weights.to_numpy()
    weights = weights / weights.mean()
    total = pd.Series(weights, index=data.index).groupby(level=0).sum()
    means = data[columns].mul(weights, axis=0).groupby(level=0).sum().div(total, axis=0)
    return means, total


@pytest.mark.parametrize("reweight", [False, True])
@pytest.mark.parametrize("weighted", WEIGHTED)
def test_between_f_statistic_against_wls(data, weighted, reweight):
    weights = weights_of(data, weighted)
    res = BetweenOLS(data["y"], data[["const", "x1", "x2"]], weights=weights).fit(
        reweight=reweight
    )
    means, total = entity_means(data, weights)
    use_total = weighted or reweight
    ref_weights = total if use_total else np.ones(total.shape[0])
    ref = sm.WLS(means["y"], means[["const", "x1", "x2"]], weights=ref_weights).fit()
    assert_allclose(res.f_statistic.stat, ref.fvalue)


def classical_equals_wald(res):
    classical = res.f_statistic
    wald = res.f_statistic_robust
    assert classical.df == wald.df
    assert classical.df_denom == wald.df_denom
    assert_allclose(classical.stat, wald.stat)


MODELS = {
    "pooled": lambda d, w: PooledOLS(d["y"], d[["const", "x1", "x2"]], weights=w),
    "between": lambda d, w: BetweenOLS(d["y"], d[["const", "x1", "x2"]], weights=w),
    "random-effects": lambda d, w: RandomEffects(
        d["y"], d[["const", "x1", "x2"]], weights=w
    ),
    "first-difference": lambda d, w: FirstDifferenceOLS(
        d["y"], d[["x1", "x2"]], weights=w
    ),
    "entity": lambda d, w: PanelOLS(
        d["y"], d[["const", "x1", "x2"]], weights=w, entity_effects=True
    ),
    "time": lambda d, w: PanelOLS(
        d["y"], d[["const", "x1", "x2"]], weights=w, time_effects=True
    ),
    "both": lambda d, w: PanelOLS(
        d["y"],
        d[["const", "x1", "x2"]],
        weights=w,
        entity_effects=True,
        time_effects=True,
    ),
    "other": lambda d, w: PanelOLS(
        d["y"], d[["const", "x1", "x2"]], weights=w, other_effects=d[["group"]]
    ),
    "entity-no-constant": lambda d, w: PanelOLS(
        d["y"], d[["x1", "x2"]], weights=w, entity_effects=True
    ),
}


@pytest.mark.parametrize("weighted", WEIGHTED)
@pytest.mark.parametrize("model", MODELS)
def test_classical_f_statistic_is_the_wald_statistic(data, model, weighted):
    # The classical F-statistic and the Wald statistic that uses the unadjusted
    # covariance test the same hypothesis in the same regression
    res = MODELS[model](data, weights_of(data, weighted)).fit(cov_type="unadjusted")
    classical_equals_wald(res)


@pytest.mark.parametrize("small_sample", [False, True])
@pytest.mark.parametrize("weighted", WEIGHTED)
def test_random_effects_f_statistic_is_the_wald_statistic(data, weighted, small_sample):
    mod = RandomEffects(
        data["y"], data[["const", "x1", "x2"]], weights=weights_of(data, weighted)
    )
    classical_equals_wald(mod.fit(small_sample=small_sample))


def test_between_reweight_f_statistic_is_the_wald_statistic(data):
    mod = BetweenOLS(data["y"], data[["const", "x1", "x2"]])
    classical_equals_wald(mod.fit(reweight=True))


@pytest.mark.parametrize("model", MODELS)
def test_weights_are_relative(data, model):
    # The weights are scaled to have a mean of one, so only the relative weights
    # are used, and these are all of the statistics
    res = MODELS[model](data, data["w"]).fit()
    scaled = MODELS[model](data, 7.3 * data["w"]).fit()
    assert_allclose(scaled.params, res.params)
    assert_allclose(scaled.cov, res.cov)
    assert_allclose(scaled.f_statistic.stat, res.f_statistic.stat)
    assert_allclose(scaled.rsquared, res.rsquared)
    assert_allclose(scaled.resid_ss, res.resid_ss)
    assert_allclose(scaled.s2, res.s2)
