import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.iv import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import BetweenOLS
from linearmodels.tests.panel._utility import (assert_results_equal,
                                               generate_data)


@pytest.fixture(params=['numpy', 'pandas', 'xarray'])
def data(request):
    return generate_data(0.0, request.param)


@pytest.fixture(params=['numpy', 'pandas', 'xarray'])
def missing_data(request):
    return generate_data(0.20, request.param)


def test_single_entity(data):
    x = data.x
    y = data.y
    if isinstance(x, pd.Panel):
        x = x.iloc[:, [0], :]
        y = y.iloc[[0], :]
    else:
        x = x[:, [0]]
        y = y[[0]]
    mod = BetweenOLS(y, x)
    res = mod.fit(reweight=True, debiased=False)

    dep = mod.dependent.dataframe
    exog = mod.exog.dataframe
    ols = IV2SLS(dep, exog, None, None)
    ols_res = ols.fit(cov_type='unadjusted')
    assert_results_equal(res, ols_res)

    res = mod.fit(cov_type='robust', debiased=False)
    ols_res = ols.fit(cov_type='robust')
    assert_results_equal(res, ols_res)

    clusters = pd.DataFrame(np.random.randint(0, 9, dep.shape),
                            index=dep.index)
    res = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    ols_res = ols.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, ols_res)


def test_single_entity_weights(data):
    x = data.x
    y = data.y
    w = data.w
    if isinstance(x, pd.Panel):
        x = x.iloc[:, [0], :]
        y = y.iloc[[0], :]
        w = w.iloc[[0], :]
    else:
        x = x[:, [0]]
        y = y[[0]]
        w = w[[0]]

    mod = BetweenOLS(y, x, weights=w)
    res = mod.fit(reweight=True, debiased=False)

    dep = mod.dependent.dataframe
    exog = mod.exog.dataframe
    ols = IV2SLS(dep, exog, None, None, weights=mod.weights.values2d)
    ols_res = ols.fit(cov_type='unadjusted')
    assert_results_equal(res, ols_res)

    res = mod.fit(cov_type='robust', debiased=False)
    ols_res = ols.fit(cov_type='robust', debiased=False)
    assert_results_equal(res, ols_res)

    clusters = pd.DataFrame(np.random.randint(0, 9, dep.shape),
                            index=dep.index)
    res = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    ols_res = ols.fit(cov_type='clustered', clusters=clusters, debiased=False)
    assert_results_equal(res, ols_res)


def test_multiple_obs_per_entity(data):
    mod = BetweenOLS(data.y, data.x)
    res = mod.fit(reweight=True, debiased=False)

    dep = mod.dependent.values3d.mean(1).T
    exog = pd.DataFrame(mod.exog.values3d.mean(1).T,
                        columns=mod.exog.vars)
    ols = IV2SLS(dep, exog, None, None)
    ols_res = ols.fit(cov_type='unadjusted')
    assert_results_equal(res, ols_res)

    res = mod.fit(cov_type='robust', debiased=False)
    ols_res = ols.fit(cov_type='robust', debiased=False)
    assert_results_equal(res, ols_res)

    clusters = mod.dependent.dataframe.copy()
    clusters.loc[:, :] = 0
    clusters = clusters.astype(np.int32)
    for entity in mod.dependent.entities:
        clusters.loc[entity] = np.random.randint(9)

    ols_clusters = PanelData(clusters).values3d.mean(1).T.astype(np.int32)
    res = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    ols_res = ols.fit(cov_type='clustered', clusters=ols_clusters)
    assert_results_equal(res, ols_res)


def test_multiple_obs_per_entity_weighted(data):
    mod = BetweenOLS(data.y, data.x, weights=data.w)
    res = mod.fit(reweight=True, debiased=False)

    weights = np.nansum(mod.weights.values3d, axis=1).T
    wdep = np.nansum(mod.weights.values3d * mod.dependent.values3d, axis=1).T
    wexog = np.nansum(mod.weights.values3d * mod.exog.values3d, axis=1).T
    wdep = wdep / weights
    wexog = wexog / weights

    dep = wdep
    exog = pd.DataFrame(wexog, columns=mod.exog.vars)

    ols = IV2SLS(dep, exog, None, None, weights=weights)
    ols_res = ols.fit(cov_type='unadjusted')
    assert_results_equal(res, ols_res)

    res = mod.fit(cov_type='robust', debiased=False)
    ols_res = ols.fit(cov_type='robust')
    assert_results_equal(res, ols_res)

    clusters = mod.dependent.dataframe.copy()
    clusters.loc[:, :] = 0
    clusters = clusters.astype(np.int32)
    for entity in mod.dependent.entities:
        clusters.loc[entity] = np.random.randint(9)

    ols_clusters = PanelData(clusters).values3d.mean(1).T.astype(np.int32)
    res = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    ols_res = ols.fit(cov_type='clustered', clusters=ols_clusters)
    assert_results_equal(res, ols_res)


def test_missing(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x)
    res = mod.fit(reweight=True, debiased=False)

    dep = mod.dependent.dataframe.groupby(level=0).mean()
    exog = mod.exog.dataframe.groupby(level=0).mean()
    weights = mod.weights.dataframe.groupby(level=0).sum()

    dep = dep.reindex(mod.dependent.entities)
    exog = exog.reindex(mod.dependent.entities)
    weights = weights.reindex(mod.dependent.entities)

    ols = IV2SLS(dep, exog, None, None, weights=weights)
    ols_res = ols.fit(cov_type='unadjusted')
    assert_results_equal(res, ols_res)

    res = mod.fit(reweight=True, cov_type='robust', debiased=False)
    ols_res = ols.fit(cov_type='robust')
    assert_results_equal(res, ols_res)

    vc1 = PanelData(missing_data.vc1)
    ols_clusters = vc1.dataframe.groupby(level=0).mean().astype(np.int32)
    ols_clusters = ols_clusters.reindex(mod.dependent.entities)

    res = mod.fit(reweight=True, cov_type='clustered', clusters=missing_data.vc1, debiased=False)
    ols_res = ols.fit(cov_type='clustered', clusters=ols_clusters)
    assert_results_equal(res, ols_res)


def test_missing_weighted(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x, weights=missing_data.w)
    res = mod.fit(reweight=True, debiased=False)

    weights = mod.weights.dataframe.groupby(level=0).sum()
    weights = weights.reindex(mod.dependent.entities)

    dep = mod.dependent.dataframe * mod.weights.dataframe.values
    dep = dep.groupby(level=0).sum()
    dep = dep.reindex(mod.dependent.entities)
    dep = dep / weights.values

    exog = mod.weights.dataframe.values * mod.exog.dataframe
    exog = exog.groupby(level=0).sum()
    exog = exog.reindex(mod.dependent.entities)
    exog = (1.0 / weights.values) * exog

    ols = IV2SLS(dep, exog, None, None, weights=weights)
    ols_res = ols.fit(cov_type='unadjusted')
    assert_results_equal(res, ols_res)


def test_unknown_covariance(data):
    mod = BetweenOLS(data.y, data.x)
    with pytest.raises(KeyError):
        mod.fit(cov_type='unknown')


def test_results_access(data):
    mod = BetweenOLS(data.y, data.x)
    res = mod.fit(debiased=False)
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()


def test_alt_rsquared(data):
    mod = BetweenOLS(data.y, data.x)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_between)


def test_alt_rsquared_missing(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_between)


def test_alt_rsquared_weighted(data):
    mod = BetweenOLS(data.y, data.x, weights=data.w)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_between)


def test_alt_rsquared_weighted_missing(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x, weights=missing_data.w)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_between)


def test_2way_cluster(data):
    mod = BetweenOLS(data.y, data.x)

    dep = mod.dependent.dataframe.groupby(level=0).mean()
    exog = mod.exog.dataframe.groupby(level=0).mean()

    clusters = mod.dependent.dataframe.copy()
    clusters.columns = ['cluster.0']
    clusters['cluster.1'] = mod.dependent.dataframe.copy()
    clusters.loc[:, :] = 0
    clusters = clusters.astype(np.int32)
    for entity in mod.dependent.entities:
        clusters.loc[entity, :] = np.random.randint(33, size=(1, 2))

    res = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)

    dep = dep.reindex(list(res.resids.index))
    exog = exog.reindex(list(res.resids.index))

    ols = IV2SLS(dep, exog, None, None)
    ols_clusters = clusters.groupby(level=0).max()
    ols_clusters = ols_clusters.reindex(list(res.resids.index))

    ols_res = ols.fit(cov_type='clustered', clusters=ols_clusters)
    assert_results_equal(res, ols_res)


def test_cluster_error(data):
    mod = BetweenOLS(data.y, data.x)
    clusters = mod.dependent.dataframe.copy()
    clusters.loc[:, :] = 0
    clusters = clusters.astype(np.int32)
    for entity in mod.dependent.entities:
        clusters.loc[entity] = np.random.randint(9)
    clusters.iloc[::7, :] = 0

    with pytest.raises(ValueError):
        mod.fit(cov_type='clustered', clusters=clusters, debiased=False)


def test_default_clusters(data):
    x = data.x
    y = data.y
    if isinstance(x, pd.Panel):
        x = x.iloc[:, [0], :]
        y = y.iloc[[0], :]
    else:
        x = x[:, [0]]
        y = y[[0]]
    mod = BetweenOLS(y, x)
    res = mod.fit(reweight=True, cov_type='clustered', debiased=False)

    dep = mod.dependent.dataframe
    exog = mod.exog.dataframe
    ols = IV2SLS(dep, exog, None, None)
    ols_res = ols.fit(cov_type='clustered')
    assert_results_equal(res, ols_res)
