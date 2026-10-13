"""
The autocorrelation robust covariance (cov_type="ac") with entities that are shorter
than the bandwidth

An entity with T_i observations has the lags 0, ..., T_i - 1. Its sum of products of
the scores is over these lags, with the weights K(j) of the kernel for the bandwidth
that was chosen: the bandwidth is not reduced to T_i - 1 for the entities that have
fewer observations than the bandwidth. The Bartlett weights of such an entity were
1 - j / T_i instead of 1 - j / (bandwidth + 1), and the quadratic spectral kernel
was computed for the bandwidth T_i - 1.

The references are

* the sums over the lags of every entity, written out in NumPy with kernels that are
  defined here;
* plm (vcovNW with the weights wj of the kernel), in
  results/ac-short-entities-reference.csv, from the panel in
  results/ac-short-entities-panel.csv, which has entities with 2 to 7 consecutive
  observations out of 12 periods. results/ac-short-entities-reference.R produced
  them;
* the cluster-HAC covariance of statsmodels (cov_nw_panel) with the same weights.

plm pairs the observations that are j calendar periods apart and the other
estimators pair the ones that are j positions apart. The panel has no gaps, so these
are the same.
"""

import os

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats import sandwich_covariance as sw

from linearmodels.panel import FirstDifferenceOLS, PanelOLS, PooledOLS
from linearmodels.panel.covariance import ACCovariance

RESULTS = os.path.join(os.path.split(os.path.abspath(__file__))[0], "results")
KERNELS = ["bartlett", "parzen", "qs"]
# The largest bandwidth is above the lengths of all of the entities
BANDWIDTHS = [1, 2, 3, 5, 8, 10]
# The covariance without the adjustments for the sample size and for the effects
RAW = {"debiased": False, "auto_df": False, "count_effects": False}


def kernel_weights(kernel, bandwidth, nlags):
    """The weights of the lags 0, ..., nlags (Andrews, 1991)"""
    lag = np.arange(nlags + 1, dtype=float)
    if kernel == "bartlett":
        z = lag / (bandwidth + 1)
        return np.where(z <= 1, 1 - z, 0.0)
    if kernel == "parzen":
        z = lag / (bandwidth + 1)
        inner = 1 - 6 * z**2 + 6 * z**3
        return np.where(z <= 0.5, inner, np.where(z <= 1, 2 * (1 - z) ** 3, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        a = 6 * np.pi * lag / (5 * bandwidth)
        weights = 3 * (np.sin(a) / a - np.cos(a)) / a**2
    weights[0] = 1.0
    return weights


def short_panel():
    panel = pd.read_csv(os.path.join(RESULTS, "ac-short-entities-panel.csv"))
    panel.insert(2, "const", 1.0)
    return panel.set_index(["entity", "time"])


def number_of_lags(kernel, bandwidth, frame):
    """
    The largest lag that is used: statsmodels needs a lag that some entity has

    The weights are those of the bandwidth. Beyond the lags that the longest entity
    has, the sums of the products are empty.
    """
    longest = int(frame.groupby(level=0).size().max()) - 1
    return longest if kernel == "qs" else min(bandwidth, longest)


def fit_model(frame, model, kernel, bandwidth):
    options = {"cov_type": "ac", "kernel": kernel, "bandwidth": bandwidth}
    if model == "pooling":
        return PooledOLS(frame["y"], frame[["const", "x1", "x2"]]).fit(
            debiased=False, **options
        )
    if model == "within":
        mod = PanelOLS(frame["y"], frame[["x1", "x2"]], entity_effects=True)
        return mod.fit(**RAW, **options)
    mod = FirstDifferenceOLS(frame["y"], frame[["x1", "x2"]])
    return mod.fit(debiased=False, **options)


def scores_by_entity(frame, model):
    """The scores x e of the regression, sorted by entity and time, and the entities"""
    if model == "pooling":
        data = frame
        exog = frame[["const", "x1", "x2"]]
        endog = frame["y"]
    elif model == "within":
        by_entity = frame.groupby(level=0)
        demeaned = frame[["y", "x1", "x2"]] - by_entity[["y", "x1", "x2"]].transform(
            "mean"
        )
        data, exog, endog = frame, demeaned[["x1", "x2"]], demeaned["y"]
    else:
        diffs = frame.groupby(level=0)[["y", "x1", "x2"]].diff().dropna()
        data, exog, endog = diffs, diffs[["x1", "x2"]], diffs["y"]
    params = np.linalg.lstsq(exog.to_numpy(), endog.to_numpy(), rcond=None)[0]
    resid = endog.to_numpy() - exog.to_numpy() @ params
    entity = data.index.get_level_values(0).to_numpy()
    return exog.to_numpy(), resid[:, None] * exog.to_numpy(), entity


def meat_by_hand(scores, entity, kernel, bandwidth):
    """The sum over the entities of the sums over the lags of the entity"""
    nvar = scores.shape[1]
    total = np.zeros((nvar, nvar))
    for key in np.unique(entity):
        xi = scores[entity == key]
        nobs = xi.shape[0]
        total += xi.T @ xi
        # Bartlett and Parzen weights are zero beyond the bandwidth, and every lag
        # that the entity has gets a weight with the quadratic spectral kernel
        last = nobs - 1 if kernel == "qs" else min(nobs - 1, int(bandwidth))
        weights = kernel_weights(kernel, bandwidth, nobs - 1)
        for lag in range(1, last + 1):
            cross = xi[lag:].T @ xi[:-lag]
            total += weights[lag] * (cross + cross.T)
    return total


@pytest.fixture(scope="module")
def frame():
    return short_panel()


@pytest.fixture(scope="module")
def plm_reference():
    reference = pd.read_csv(os.path.join(RESULTS, "ac-short-entities-reference.csv"))
    out = {}
    for key, group in reference.groupby(["model", "kernel", "bandwidth"]):
        out[key] = group.pivot(index="row", columns="column", values="value").to_numpy()
    return out


def test_panel_has_short_entities(frame):
    lengths = frame.groupby(level=0).size()
    assert lengths.min() == 2
    assert lengths.max() == 7
    # Most entities are shorter than the bandwidths that are used
    assert (lengths <= 3).sum() > lengths.shape[0] / 2


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("model", ["pooling", "within", "fd"])
def test_ac_covariance_short_entities_against_plm(
    frame, plm_reference, model, kernel, bandwidth
):
    res = fit_model(frame, model, kernel, bandwidth)
    expected = plm_reference[(model, kernel, bandwidth)]
    assert res.cov.shape == expected.shape
    assert_allclose(res.cov.to_numpy(), expected, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("model", ["pooling", "within", "fd"])
def test_ac_covariance_short_entities_by_hand(frame, model, kernel, bandwidth):
    res = fit_model(frame, model, kernel, bandwidth)
    exog, scores, entity = scores_by_entity(frame, model)
    bread = np.linalg.inv(exog.T @ exog)
    meat = meat_by_hand(scores, entity, kernel, bandwidth)
    assert_allclose(res.cov.to_numpy(), bread @ meat @ bread, rtol=1e-8)


def statsmodels_covariance(frame, model, kernel, bandwidth):
    """The cluster-HAC covariance of statsmodels, with the weights of the kernel"""
    if model == "pooling":
        exog, endog, names = frame[["const", "x1", "x2"]], frame["y"], None
        data = frame
    elif model == "within":
        entity = frame.index.get_level_values(0)
        dummies = pd.get_dummies(entity, dtype=float).set_axis(frame.index)
        exog = pd.concat([frame[["x1", "x2"]], dummies], axis=1)
        endog, names, data = frame["y"], ["x1", "x2"], frame
    else:
        diffs = frame.groupby(level=0)[["y", "x1", "x2"]].diff().dropna()
        exog, endog, names, data = diffs[["x1", "x2"]], diffs["y"], None, diffs
    res = sm.OLS(endog, exog).fit()
    entity = data.index.get_level_values(0).to_numpy()
    starts = np.flatnonzero(np.r_[True, entity[1:] != entity[:-1]])
    groups = list(zip(starts, np.r_[starts[1:], entity.shape[0]], strict=True))
    nlags = number_of_lags(kernel, bandwidth, data)
    cov = sw.cov_nw_panel(
        res,
        nlags,
        groups,
        weights_func=lambda n: kernel_weights(kernel, bandwidth, n),
        use_correction=False,
    )
    if names is not None:
        cov = pd.DataFrame(cov, index=exog.columns, columns=exog.columns)
        cov = cov.loc[names, names].to_numpy()
    return cov


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("model", ["pooling", "within", "fd"])
def test_ac_covariance_short_entities_against_statsmodels(
    frame, model, kernel, bandwidth
):
    res = fit_model(frame, model, kernel, bandwidth)
    expected = statsmodels_covariance(frame, model, kernel, bandwidth)
    assert_allclose(res.cov.to_numpy(), expected, rtol=1e-8)


@pytest.mark.parametrize("kernel", KERNELS)
def test_ac_covariance_of_a_panel_of_short_entities_by_hand(frame, kernel):
    # Every entity has at most 3 observations, so that every entity is shorter than
    # the bandwidths below 3
    lengths = frame.groupby(level=0).size()
    short = frame.loc[lengths[lengths <= 3].index]
    for bandwidth in (2, 5, 10):
        res = fit_model(short, "pooling", kernel, bandwidth)
        exog, scores, entity = scores_by_entity(short, "pooling")
        bread = np.linalg.inv(exog.T @ exog)
        meat = meat_by_hand(scores, entity, kernel, bandwidth)
        assert_allclose(res.cov.to_numpy(), bread @ meat @ bread, rtol=1e-8)


def test_bartlett_weights_of_a_short_entity_are_nominal():
    # One entity with 3 observations, bandwidth 4: the weights of the lags 1 and 2 are
    # 1 - 1/5 and 1 - 2/5, not 1 - 1/3 and 1 - 2/3
    rg = np.random.default_rng(5)
    nobs = 3
    x = rg.standard_normal((nobs, 1))
    y = x + rg.standard_normal((nobs, 1))
    params = np.array([[0.7]])
    ids = np.zeros((nobs, 1), dtype=np.int64)
    time_ids = np.arange(nobs, dtype=np.int64)[:, None]
    cov = ACCovariance(
        y, x, params, ids, time_ids, kernel="bartlett", bandwidth=4, debiased=False
    ).cov
    xi = (x * (y - x @ params)).ravel()
    meat = xi @ xi
    for lag, weight in ((1, 1 - 1 / 5), (2, 1 - 2 / 5)):
        meat += 2 * weight * (xi[lag:] @ xi[:-lag])
    assert_allclose(cov, meat / (x.T @ x) ** 2, rtol=1e-12)
