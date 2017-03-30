import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.util.testing import assert_series_equal, assert_frame_equal

from linearmodels.iv import IV2SLS
from linearmodels.panel.model import BetweenOLS
from linearmodels.tests.panel._utility import generate_data


@pytest.fixture(params=['numpy', 'pandas', 'xarray'])
def data(request):
    return generate_data(0.0, request.param)


@pytest.fixture(params=['numpy', 'pandas', 'xarray'])
def missing_data(request):
    return generate_data(0.20, request.param)


def assert_results_equal(res1, res2):
    assert_series_equal(res1.params, res2.params)
    assert_series_equal(res1.pvalues, res2.pvalues)
    assert_series_equal(res1.tstats, res2.tstats)
    assert_frame_equal(res1.cov, res2.cov)
    assert_frame_equal(res1.conf_int(), res2.conf_int())
    assert_allclose(res1.rsquared, res2.rsquared)
    assert_allclose(res1.total_ss, res2.total_ss)
    assert_allclose(res1.resid_ss, res2.resid_ss)
    assert_allclose(res1.model_ss, res2.model_ss)
    assert_allclose(res1.s2, res2.s2)
    assert_allclose(res1.df_model, res2.df_model)
    assert_allclose(res1.df_resid, res2.df_resid)


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
    
    dep = np.nanmean(mod.dependent.values3d, axis=1).T
    exog = pd.DataFrame(np.nanmean(mod.exog.values3d, axis=1).T,
                        columns=mod.exog.vars)
    weights = np.nansum(mod.weights.values3d, axis=1).T
    ols = IV2SLS(dep, exog, None, None, weights=weights)
    ols_res = ols.fit('unadjusted')
    assert_results_equal(res, ols_res)


def test_missing_weighted(missing_data):
    mod = BetweenOLS(missing_data.y, missing_data.x, weights=missing_data.w)
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
