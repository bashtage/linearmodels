"""
The autocorrelation robust covariance (cov_type="ac") of the panel models

The estimator is the sandwich

    (X'X)^-1 [ sum_i sum_{j=-L}^{L} w_j sum_t xi_it xi_i,t-j' ] (X'X)^-1

with the scores xi_it = x_it e_it and the Bartlett weights w_j = 1 - |j| / (L + 1).
Each entity contributes its sum of the products of the scores, and an entity with more
observations has more weight. With L = 0 it is White's covariance. It was the average
of the sums of the entities that were divided by their numbers of observations, which
is the same only if the panel is balanced.

The references are

* plm (vcovNW), in results/ac-covariance-reference.csv, from the panel in
  results/ac-covariance-panel.csv, which has entities that enter and leave at different
  periods and no gaps. results/ac-covariance-reference.R produced them;
* the covariance with the cluster-HAC option of statsmodels ("hac-panel");
* the covariance without lags, which is White's.

If an entity has gaps, plm pairs the observations that are a number of calendar periods
apart, and statsmodels and linearmodels pair the observations that are a number of
positions apart in the observations of the entity. The panels with gaps are only compared
with statsmodels.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.api as sm

from linearmodels.panel import PanelOLS, PooledOLS

RESULTS = os.path.join(os.path.split(os.path.abspath(__file__))[0], "results")
MAXLAGS = [0, 1, 2, 3]
# The covariance without the adjustments for the sample size and for the effects
RAW = {"debiased": False, "kernel": "bartlett"}


def plm_panel():
    panel = pd.read_csv(os.path.join(RESULTS, "ac-covariance-panel.csv"))
    panel.insert(2, "const", 1.0)
    return panel.set_index(["entity", "time"])


def random_panel(balance, seed=0, nentity=30, nperiod=10):
    """A panel that is balanced or whose entities have missing periods (gaps)"""
    rg = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [np.arange(nentity), np.arange(nperiod)], names=["entity", "time"]
    )
    x1 = rg.standard_normal(index.shape[0]) + 0.3 * index.codes[0] % 4
    x2 = rg.standard_normal(index.shape[0])
    shocks = rg.standard_normal((nentity, nperiod))
    for t in range(1, nperiod):
        shocks[:, t] += 0.5 * shocks[:, t - 1]
    effect = rg.standard_normal(nentity)[index.codes[0]]
    y = 1 + 0.5 * x1 - 0.3 * x2 + effect + (0.5 + np.abs(x1)) * shocks.ravel()
    frame = pd.DataFrame({"y": y, "const": 1.0, "x1": x1, "x2": x2}, index=index)
    if balance == "gaps":
        keep = rg.random((nentity, nperiod)) > 0.3
        for i in range(nentity):
            keep[i, rg.choice(nperiod, 4, replace=False)] = True
        frame = frame.loc[keep.ravel()]
    return frame


def fit_model(frame, model, maxlag, **fit_options):
    """The pooled model or the model with entity effects, with an AC covariance"""
    options = {**RAW, "cov_type": "ac", "bandwidth": maxlag, **fit_options}
    if model == "pooling":
        return PooledOLS(frame["y"], frame[["const", "x1", "x2"]]).fit(**options)
    mod = PanelOLS(frame["y"], frame[["x1", "x2"]], entity_effects=True)
    return mod.fit(auto_df=False, count_effects=False, **options)


def hac_panel(frame, model, maxlag):
    """The cluster-HAC covariance of statsmodels for the coefficients of the model"""
    exog = frame[["const", "x1", "x2"]]
    if model == "within":
        entity = frame.index.get_level_values(0)
        exog = pd.concat(
            [
                frame[["x1", "x2"]],
                pd.get_dummies(entity).astype(float).set_axis(frame.index),
            ],
            axis=1,
        )
    codes = pd.factorize(frame.index.get_level_values(0))[0]
    res = sm.OLS(frame["y"], exog).fit(
        cov_type="hac-panel",
        cov_kwds={
            "groups": codes,
            "maxlags": maxlag,
            "kernel": "bartlett",
            "use_correction": False,
        },
    )
    names = list(frame[["const", "x1", "x2"]] if model == "pooling" else ["x1", "x2"])
    return res.cov_params().loc[names, names].to_numpy()


@pytest.fixture(scope="module")
def plm_reference():
    reference = pd.read_csv(os.path.join(RESULTS, "ac-covariance-reference.csv"))
    out = {}
    for (model, maxlag), group in reference.groupby(["model", "maxlag"]):
        out[(model, maxlag)] = group.pivot(
            index="row", columns="column", values="value"
        ).to_numpy()
    return out


@pytest.mark.parametrize("maxlag", MAXLAGS)
@pytest.mark.parametrize("model", ["pooling", "within"])
def test_ac_covariance_against_plm(plm_reference, model, maxlag):
    res = fit_model(plm_panel(), model, maxlag)
    expected = plm_reference[(model, maxlag)]
    # The terms are alphabetical in the pivot: (Intercept), x1, x2 and x1, x2
    assert res.cov.shape == expected.shape
    assert_allclose(res.cov.to_numpy(), expected, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("maxlag", MAXLAGS)
@pytest.mark.parametrize("model", ["pooling", "within"])
@pytest.mark.parametrize("balance", ["balanced", "gaps", "plm"])
def test_ac_covariance_against_statsmodels(balance, model, maxlag):
    frame = plm_panel() if balance == "plm" else random_panel(balance)
    res = fit_model(frame, model, maxlag)
    assert_allclose(res.cov.to_numpy(), hac_panel(frame, model, maxlag), rtol=1e-8)


@pytest.mark.parametrize("balance", ["balanced", "gaps", "plm"])
def test_ac_covariance_without_lags_is_the_robust_covariance(balance):
    frame = plm_panel() if balance == "plm" else random_panel(balance)
    ac = fit_model(frame, "pooling", 0)
    robust = PooledOLS(frame["y"], frame[["const", "x1", "x2"]]).fit(
        cov_type="robust", debiased=False
    )
    assert_allclose(ac.cov.to_numpy(), robust.cov.to_numpy(), rtol=1e-8)


@pytest.mark.parametrize("balance", ["balanced", "gaps"])
def test_ac_covariance_debiased(balance):
    # The sandwich is multiplied by n / (n - k) if debiased
    frame = random_panel(balance)
    raw = fit_model(frame, "pooling", 2)
    debiased = fit_model(frame, "pooling", 2, debiased=True)
    nobs, nvar = frame.shape[0], 3
    assert_allclose(debiased.cov.to_numpy(), raw.cov.to_numpy() * nobs / (nobs - nvar))
