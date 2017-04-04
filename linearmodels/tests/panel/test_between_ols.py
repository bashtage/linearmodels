import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.iv import IV2SLS
from linearmodels.panel.model import BetweenOLS
from linearmodels.tests.panel._utility import assert_results_equal, generate_data


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
    res = mod.fit(reweight=True)
    
    dep = mod.dependent.dataframe
    exog = mod.exog.dataframe
    ols = IV2SLS(dep, exog, None, None)
    ols_res = ols.fit('unadjusted')
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
    res = mod.fit(reweight=True)
    
    dep = mod.dependent.dataframe
    exog = mod.exog.dataframe
    ols = IV2SLS(dep, exog, None, None, weights=mod.weights.values2d)
    ols_res = ols.fit('unadjusted')
    assert_results_equal(res, ols_res)


def test_multiple_obs_per_entity(data):
    mod = BetweenOLS(data.y, data.x)
    res = mod.fit(reweight=True)
    
    dep = mod.dependent.values3d.mean(1).T
    exog = pd.DataFrame(mod.exog.values3d.mean(1).T,
                        columns=mod.exog.vars)
    ols = IV2SLS(dep, exog, None, None)
    ols_res = ols.fit('unadjusted')
    assert_results_equal(res, ols_res)


def test_multiple_obs_per_entity_weighted(data):
    mod = BetweenOLS(data.y, data.x, weights=data.w)
    res = mod.fit(reweight=True)
    
    weights = np.nansum(mod.weights.values3d, axis=1).T
    wdep = np.nansum(mod.weights.values3d * mod.dependent.values3d, axis=1).T
    wexog = np.nansum(mod.weights.values3d * mod.exog.values3d, axis=1).T
    wdep = wdep / weights
    wexog = wexog / weights
    
    dep = wdep
    exog = pd.DataFrame(wexog, columns=mod.exog.vars)
    
    ols = IV2SLS(dep, exog, None, None, weights=weights)
    ols_res = ols.fit('unadjusted')
    assert_results_equal(res, ols_res)


def test_missing(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x)
    res = mod.fit(reweight=True)
    
    dep = mod.dependent.dataframe.groupby(level=0).mean()
    exog = mod.exog.dataframe.groupby(level=0).mean()
    weights = mod.weights.dataframe.groupby(level=0).sum()
    
    dep = dep.reindex(mod.dependent.entities)
    exog = exog .reindex(mod.dependent.entities)
    weights = weights.reindex(mod.dependent.entities)
    
    ols = IV2SLS(dep, exog, None, None, weights=weights)
    ols_res = ols.fit('unadjusted')
    assert_results_equal(res, ols_res)


def test_missing_weighted(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x, weights=missing_data.w)
    res = mod.fit(reweight=True)

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
    ols_res = ols.fit('unadjusted')
    assert_results_equal(res, ols_res)


def test_unsupported_covariance(data):
    mod = BetweenOLS(data.y, data.x)
    with pytest.raises(ValueError):
        mod.fit(cov_type='clustered', clusters=None)


def test_unknown_covariance(data):
    mod = BetweenOLS(data.y, data.x)
    with pytest.raises(KeyError):
        mod.fit(cov_type='unknown')


def test_results_access(data):
    mod = BetweenOLS(data.y, data.x)
    res = mod.fit()
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()


def test_alt_rsquared(data):
    mod = BetweenOLS(data.y, data.x)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_between)


def test_alt_rsquared_missing(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_between)


def test_alt_rsquared_weighted(data):
    mod = BetweenOLS(data.y, data.x, weights=data.w)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_between)


def test_alt_rsquared_weighted_missing(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x, weights=missing_data.w)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_between)
