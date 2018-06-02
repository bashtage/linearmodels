from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.iv import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PooledOLS
from linearmodels.tests.panel._utility import (assert_results_equal, datatypes, generate_data,
                                               assert_frame_similar)

pytestmark = pytest.mark.filterwarnings('ignore::linearmodels.utility.MissingValueWarning')

missing = [0.0, 0.20]
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(missing, datatype, const=const, other_effects=1)


def test_pooled_ols(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit(debiased=False)

    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    y.index = np.arange(len(y))
    x.index = y.index

    res2 = IV2SLS(y, x, None, None).fit(cov_type='unadjusted')
    assert_results_equal(res, res2)

    res3 = mod.fit(cov_type='homoskedastic', debiased=False)
    assert_results_equal(res, res3)


def test_pooled_ols_weighted(data):
    mod = PooledOLS(data.y, data.x, weights=data.w)
    res = mod.fit(debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    y.index = np.arange(len(y))
    w.index = x.index = y.index

    res2 = IV2SLS(y, x, None, None, weights=w).fit(cov_type='unadjusted')
    assert_results_equal(res, res2)


def test_diff_data_size(data):
    if isinstance(data.x, pd.DataFrame):
        entities = data.x.index.levels[0]
        x = data.x.loc[pd.IndexSlice[entities[0]:entities[-2]]]
        y = data.y
    elif isinstance(data.x, np.ndarray):
        x = data.x
        y = data.y[:-1]
    else:  # xr.DataArray:
        x = data.x[:, :, :-1]
        y = data.y

    with pytest.raises(ValueError):
        PooledOLS(y, x)


def test_rank_deficient_array(data):
    x = data.x.copy()
    if isinstance(data.x, pd.DataFrame):
        x.iloc[:, 1] = x.iloc[:, 0]
    else:
        x[1] = x[0]
    with pytest.raises(ValueError):
        PooledOLS(data.y, x)


def test_results_access(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit(debiased=False)
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()

    mod = PooledOLS(data.y, data.x)
    res = mod.fit(debiased=True)
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()

    if not isinstance(data.x, pd.DataFrame):
        return
    x = data.y.copy()
    x.iloc[:, :] = 1
    mod = PooledOLS(data.y, x)
    res = mod.fit(debiased=False)
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()


def test_alt_rsquared(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_overall)


def test_alt_rsquared_weighted(data):
    mod = PooledOLS(data.y, data.x, weights=data.w)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_overall)


def test_cov_equiv(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit(cov_type='robust', debiased=False)
    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    y.index = np.arange(len(y))
    x.index = y.index
    res2 = IV2SLS(y, x, None, None).fit(cov_type='robust')
    assert_results_equal(res, res2)

    res3 = mod.fit(cov_type='heteroskedastic', debiased=False)
    assert_results_equal(res, res3)


def test_cov_equiv_weighted(data):
    mod = PooledOLS(data.y, data.x, weights=data.w)
    res = mod.fit(cov_type='robust', debiased=False)
    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    w = mod.weights.dataframe.copy()
    y.index = np.arange(len(y))
    w.index = x.index = y.index

    res2 = IV2SLS(y, x, None, None, weights=w).fit(cov_type='robust')
    assert_results_equal(res, res2)

    res3 = mod.fit(cov_type='heteroskedastic', debiased=False)
    assert_results_equal(res, res3)


def test_cov_equiv_cluster(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit(cov_type='clustered', cluster_entity=True, debiased=False)
    y = PanelData(data.y)
    clusters = pd.DataFrame(y.entity_ids, index=y.index)
    res2 = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    assert_results_equal(res, res2)

    res = mod.fit(cov_type='clustered', cluster_time=True, debiased=False)
    clusters = pd.DataFrame(y.time_ids, index=y.index)
    res2 = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    assert_results_equal(res, res2)

    res = mod.fit(cov_type='clustered', clusters=data.vc1, debiased=False)
    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    y.index = np.arange(len(y))
    x.index = y.index
    clusters = mod.reformat_clusters(data.vc1)
    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters.dataframe,
                       debiased=False)
    assert_results_equal(res, res2)


def test_cov_equiv_cluster_weighted(data):
    mod = PooledOLS(data.y, data.x, weights=data.w)
    res = mod.fit(cov_type='clustered', clusters=data.vc1, debiased=False)

    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    w = mod.weights.dataframe
    y.index = np.arange(len(y))
    w.index = x.index = y.index
    clusters = mod.reformat_clusters(data.vc1)
    ols_mod = IV2SLS(y, x, None, None, weights=w)
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters.dataframe)
    assert_results_equal(res, res2)


def test_two_way_clustering(data):
    mod = PooledOLS(data.y, data.x)

    y = PanelData(data.y)
    entity_clusters = pd.DataFrame(y.entity_ids, index=y.index)
    vc1 = PanelData(data.vc1)
    clusters = vc1.copy()
    clusters.dataframe['var.cluster.entity'] = entity_clusters
    clusters._frame = clusters._frame.astype(np.int64)
    res = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)

    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    y.index = np.arange(len(y))
    x.index = y.index
    clusters = mod.reformat_clusters(clusters)

    ols_mod = IV2SLS(y, x, None, None)
    ols_res = ols_mod.fit(cov_type='clustered', clusters=clusters.dataframe)
    assert_results_equal(res, ols_res)


def test_fitted_effects_residuals(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit()
    expected = pd.DataFrame(res.resids.copy())
    expected.columns = ['idiosyncratic']
    assert_allclose(res.idiosyncratic, expected)
    assert_frame_similar(res.idiosyncratic, expected)

    expected = mod.dependent.values2d - res.resids.values[:, None]
    expected = pd.DataFrame(expected, index=res.resids.index, columns=['fitted_values'])
    assert_allclose(res.fitted_values, expected)
    assert_frame_similar(res.fitted_values, expected)

    expected.iloc[:, 0] = np.nan
    expected.columns = ['estimated_effects']
    assert_allclose(res.estimated_effects, expected)
    assert_frame_similar(res.estimated_effects, expected)
