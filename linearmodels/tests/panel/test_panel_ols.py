from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.iv.model import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS, PooledOLS
from linearmodels.tests.panel._utility import (assert_results_equal,
                                               generate_data)
from linearmodels.utility import AttrDict

missing = [0.0, 0.02, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(missing, datatype, const=const, ntk=(91, 7, 5), other_effects=2)


perms = list(product(missing, datatypes))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def const_data(request):
    missing, datatype = request.param
    data = generate_data(missing, datatype, ntk=(91, 7, 1))
    y = PanelData(data.y).dataframe
    x = y.copy()
    x.iloc[:, :] = 1
    x.columns = ['Const']
    return AttrDict(y=y, x=x, w=PanelData(data.w).dataframe)


def test_const_data_only(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x)
    res = mod.fit(debiased=False)
    res2 = IV2SLS(y, x, None, None).fit()
    assert_allclose(res.params, res2.params)


def test_const_data_only_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, weights=const_data.w)
    res = mod.fit(debiased=False)
    res2 = IV2SLS(y, x, None, None, weights=const_data.w).fit()
    assert_allclose(res.params, res2.params)


def test_const_data_entity(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effects=True)
    res = mod.fit(debiased=False)

    x = mod.exog.dataframe
    d = mod.dependent.dummies('entity', drop_first=True)
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_entity_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effects=True, weights=const_data.w)
    res = mod.fit(debiased=False)

    y = mod.dependent.dataframe
    w = mod.weights.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies('entity', drop_first=True)
    d_columns = list(d.columns)

    root_w = np.sqrt(w.values)
    z = np.ones_like(x)
    wd = root_w * d.values
    wz = root_w
    d = d - z @ np.linalg.lstsq(wz, wd)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + d_columns)

    res2 = IV2SLS(y, xd, None, None, weights=w).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_time(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, time_effects=True)
    res = mod.fit(debiased=False)

    x = mod.exog.dataframe
    d = mod.dependent.dummies('time', drop_first=True)
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_time_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, time_effects=True, weights=const_data.w)
    res = mod.fit(debiased=False)

    y = mod.dependent.dataframe
    w = mod.weights.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies('time', drop_first=True)
    d_columns = list(d.columns)

    root_w = np.sqrt(w.values)
    z = np.ones_like(x)
    wd = root_w * d.values
    wz = root_w
    d = d - z @ np.linalg.lstsq(wz, wd)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + d_columns)

    res2 = IV2SLS(y, xd, None, None, weights=w).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_both(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    res = mod.fit(debiased=False)

    x = mod.exog.dataframe
    d1 = mod.dependent.dummies('entity', drop_first=True)
    d2 = mod.dependent.dummies('time', drop_first=True)
    d = np.c_[d1.values, d2.values]
    d = pd.DataFrame(d, index=x.index, columns=list(d1.columns) + list(d2.columns))
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_both_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True, weights=const_data.w)
    res = mod.fit(debiased=False)

    w = mod.weights.dataframe
    x = mod.exog.dataframe

    d1 = mod.dependent.dummies('entity', drop_first=True)
    d2 = mod.dependent.dummies('time', drop_first=True)
    d = np.c_[d1.values, d2.values]
    root_w = np.sqrt(w.values)
    z = np.ones_like(x)
    wd = root_w * d
    wz = root_w
    d = d - z @ np.linalg.lstsq(wz, wd)[0]
    d = pd.DataFrame(d, index=x.index, columns=list(d1.columns) + list(d2.columns))

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None, weights=w).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_panel_no_effects(data):
    res = PanelOLS(data.y, data.x).fit()
    res2 = PooledOLS(data.y, data.x).fit()
    assert_results_equal(res, res2)


def test_panel_no_effects_weighted(data):
    res = PanelOLS(data.y, data.x, weights=data.w).fit()
    res2 = PooledOLS(data.y, data.x, weights=data.w).fit()
    assert_results_equal(res, res2)


def test_panel_entity_lsdv(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    if mod.has_constant:
        d = mod.dependent.dummies('entity', drop_first=True)
        z = np.ones_like(y)
        d_demean = d.values - z @ np.linalg.lstsq(z, d.values)[0]
    else:
        d = mod.dependent.dummies('entity', drop_first=False)
        d_demean = d.values

    xd = np.c_[x.values, d_demean]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted', debiased=False)
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(data.vc1)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(data.vc2)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_time=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.time_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_entity=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.entity_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_entity_fwl(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    if mod.has_constant:
        d = mod.dependent.dummies('entity', drop_first=True)
        z = np.ones_like(y)
        d_demean = d.values - z @ np.linalg.lstsq(z, d.values)[0]
    else:
        d = mod.dependent.dummies('entity', drop_first=False)
        d_demean = d.values

    x = x - d_demean @ np.linalg.lstsq(d_demean, x)[0]
    y = y - d_demean @ np.linalg.lstsq(d_demean, y)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_df=False)


def test_panel_time_lsdv(data):
    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies('time', drop_first=mod.has_constant)
    d_cols = list(d.columns)
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ np.linalg.lstsq(z, d)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + d_cols)

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_time=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.time_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_entity=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.entity_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_time_fwl(data):
    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies('time', drop_first=mod.has_constant)
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ np.linalg.lstsq(z, d)[0]

    x = x - d @ np.linalg.lstsq(d, x)[0]
    y = y - d @ np.linalg.lstsq(d, y)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_df=False)


def test_panel_both_lsdv(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d1 = mod.dependent.dummies('entity', drop_first=mod.has_constant)
    d2 = mod.dependent.dummies('time', drop_first=True)
    d = np.c_[d1.values, d2.values]

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ np.linalg.lstsq(z, d)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd,
                      index=x.index,
                      columns=list(x.columns) + list(d1.columns) + list(d2.columns))

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_time=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.time_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_entity=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.entity_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_both_fwl(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    d1 = mod.dependent.dummies('entity', drop_first=mod.has_constant)
    d2 = mod.dependent.dummies('time', drop_first=True)
    d = np.c_[d1.values, d2.values]

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ np.linalg.lstsq(z, d)[0]

    x = x - d @ np.linalg.lstsq(d, x)[0]
    y = y - d @ np.linalg.lstsq(d, y)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_df=False)


def test_panel_entity_lsdv_weighted(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, weights=data.w)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    d = mod.dependent.dummies('entity', drop_first=mod.has_constant)
    d_cols = d.columns
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        root_w = np.sqrt(w.values)
        wd = root_w * d
        wz = root_w * z
        d = d - z @ np.linalg.lstsq(wz, wd)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d_cols))

    ols_mod = IV2SLS(y, xd, None, None, weights=w)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_time=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.time_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_entity=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.entity_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_time_lsdv_weighted(data):
    mod = PanelOLS(data.y, data.x, time_effects=True, weights=data.w)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    d = mod.dependent.dummies('time', drop_first=mod.has_constant)
    d_cols = d.columns
    d = d.values
    if mod.has_constant:
        z = np.ones_like(y)
        root_w = np.sqrt(w.values)
        wd = root_w * d
        wz = root_w * z
        d = d - z @ np.linalg.lstsq(wz, wd)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d_cols))

    ols_mod = IV2SLS(y, xd, None, None, weights=w)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_time=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.time_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_entity=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.entity_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_both_lsdv_weighted(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True, weights=data.w)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    d1 = mod.dependent.dummies('entity', drop_first=mod.has_constant)
    d2 = mod.dependent.dummies('time', drop_first=True)
    d = np.c_[d1.values, d2.values]

    if mod.has_constant:
        z = np.ones_like(y)
        root_w = np.sqrt(w.values)
        wd = root_w * d
        wz = root_w * z
        d = d - z @ np.linalg.lstsq(wz, wd)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd,
                      index=x.index,
                      columns=list(x.columns) + list(d1.columns) + list(d2.columns))

    ols_mod = IV2SLS(y, xd, None, None, weights=w)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_fit=False)
    assert_allclose(res.rsquared_inclusive, res2.rsquared)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_time=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.time_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_entity=True, auto_df=False, count_effects=False,
                  debiased=False)
    clusters = pd.DataFrame(mod.dependent.entity_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_entity_other_equivalence(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit()
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    cats = pd.DataFrame(mod.dependent.entity_ids, index=mod.dependent.index)

    mod2 = PanelOLS(y, x, other_effects=cats)
    res2 = mod2.fit()
    assert_results_equal(res, res2)


def test_panel_time_other_equivalence(data):
    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit()
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    cats = pd.DataFrame(mod.dependent.time_ids, index=mod.dependent.index)

    mod2 = PanelOLS(y, x, other_effects=cats)
    res2 = mod2.fit()
    assert_results_equal(res, res2)


def test_panel_entity_time_other_equivalence(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit()
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    c1 = mod.dependent.entity_ids
    c2 = mod.dependent.time_ids
    cats = np.c_[c1, c2]
    cats = pd.DataFrame(cats, index=mod.dependent.index)

    mod2 = PanelOLS(y, x, other_effects=cats)
    res2 = mod2.fit()
    assert_results_equal(res, res2)


def test_panel_other_lsdv(data):
    mod = PanelOLS(data.y, data.x, other_effects=data.c)
    assert 'Num Other Effects: 2' in str(mod)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    c = mod._other_effect_cats.dataframe.copy()
    d = []
    d_columns = []
    for i, col in enumerate(c):
        s = c[col].copy()
        dummies = pd.get_dummies(s.astype(np.int64), drop_first=(mod.has_constant or i > 0))
        dummies.columns = [s.name + '_val_' + str(c) for c in dummies.columns]
        d_columns.extend(list(dummies.columns))
        d.append(dummies.values)
    d = np.column_stack(d)

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ np.linalg.lstsq(z, d)[0]

    xd = np.c_[x.values, d]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d_columns))

    ols_mod = IV2SLS(y, xd, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_fit=False)

    res3 = mod.fit(cov_type='unadjusted', auto_df=False, count_effects=False, debiased=False)
    assert_results_equal(res, res3)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc1
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False, count_effects=False,
                  debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    clusters = data.vc2
    ols_clusters = mod.reformat_clusters(clusters)
    res = mod.fit(cov_type='clustered', clusters=clusters, auto_df=False,
                  count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='clustered', clusters=ols_clusters.dataframe)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_time=True, auto_df=False,
                  count_effects=False, debiased=False)
    clusters = pd.DataFrame(mod.dependent.time_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)

    res = mod.fit(cov_type='clustered', cluster_entity=True, auto_df=False,
                  count_effects=False, debiased=False)
    clusters = pd.DataFrame(mod.dependent.entity_ids,
                            index=mod.dependent.index,
                            columns=['var.clust'])
    res2 = ols_mod.fit(cov_type='clustered', clusters=clusters)
    assert_results_equal(res, res2, test_fit=False)


def test_panel_other_fwl(data):
    mod = PanelOLS(data.y, data.x, other_effects=data.c)
    res = mod.fit(auto_df=False, count_effects=False, debiased=False)

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    c = mod._other_effect_cats.dataframe
    d = []
    d_columns = []
    for i, col in enumerate(c):
        s = c[col].copy()
        dummies = pd.get_dummies(s.astype(np.int64), drop_first=(mod.has_constant or i > 0))
        dummies.columns = [s.name + '_val_' + str(c) for c in dummies.columns]
        d_columns.extend(list(dummies.columns))
        d.append(dummies.values)
    d = np.column_stack(d)

    if mod.has_constant:
        z = np.ones_like(y)
        d = d - z @ np.linalg.lstsq(z, d)[0]

    x = x - d @ np.linalg.lstsq(d, x)[0]
    y = y - d @ np.linalg.lstsq(d, y)[0]

    ols_mod = IV2SLS(y, x, None, None)
    res2 = ols_mod.fit(cov_type='unadjusted')
    assert_results_equal(res, res2, test_df=False)

    res = mod.fit(cov_type='robust', auto_df=False, count_effects=False, debiased=False)
    res2 = ols_mod.fit(cov_type='robust')
    assert_results_equal(res, res2, test_df=False)


def test_panel_other_incorrect_size(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    cats = pd.DataFrame(mod.dependent.entity_ids, index=mod.dependent.index)
    cats = PanelData(cats)
    cats = cats.dataframe.iloc[:cats.dataframe.shape[0] // 2, :]

    with pytest.raises(ValueError):
        PanelOLS(y, x, other_effects=cats)


def test_results_access(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit()
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()

    mod = PanelOLS(data.y, data.x, other_effects=data.c)
    res = mod.fit()
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()

    mod = PanelOLS(data.y, data.x, time_effects=True, entity_effects=True)
    res = mod.fit()
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()

    mod = PanelOLS(data.y, data.x)
    res = mod.fit()
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()

    const = PanelData(data.y).copy()
    const.dataframe.iloc[:, :] = 1
    const.dataframe.columns = ['const']
    mod = PanelOLS(data.y, const)
    res = mod.fit()
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()


def test_alt_rsquared(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_within)


def test_alt_rsquared_weighted(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True, weights=data.w)
    res = mod.fit(debiased=False)
    assert_allclose(res.rsquared, res.rsquared_within)


def test_too_many_effects(data):
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, entity_effects=True, time_effects=True, other_effects=data.c)


def test_cov_equiv_cluster(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True, debiased=False)

    y = PanelData(data.y)
    clusters = pd.DataFrame(y.entity_ids, index=y.index)
    res2 = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    assert_results_equal(res, res2)

    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit(cov_type='clustered', cluster_time=True, debiased=False)
    y = PanelData(data.y)
    clusters = pd.DataFrame(y.time_ids, index=y.index)
    res2 = mod.fit(cov_type='clustered', clusters=clusters, debiased=False)
    assert_results_equal(res, res2)

    res = mod.fit(cov_type='clustered', debiased=False)
    res2 = mod.fit(cov_type='clustered', clusters=None, debiased=False)
    assert_results_equal(res, res2)


def test_cluster_smoke(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    mod.fit(cov_type='clustered', cluster_time=True, debiased=False)
    mod.fit(cov_type='clustered', cluster_entity=True, debiased=False)
    c2 = PanelData(data.vc2)
    c1 = PanelData(data.vc1)

    mod.fit(cov_type='clustered', clusters=c2, debiased=False)
    mod.fit(cov_type='clustered', cluster_entity=True, clusters=c1, debiased=False)
    mod.fit(cov_type='clustered', cluster_time=True, clusters=c1, debiased=False)
    with pytest.raises(ValueError):
        mod.fit(cov_type='clustered', cluster_time=True, clusters=c2, debiased=False)
    with pytest.raises(ValueError):
        mod.fit(cov_type='clustered', cluster_entity=True, clusters=c2, debiased=False)
    with pytest.raises(ValueError):
        mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True, clusters=c1,
                debiased=False)
    with pytest.raises(ValueError):
        clusters = c1.dataframe.iloc[:c1.dataframe.shape[0] // 2]
        mod.fit(cov_type='clustered', clusters=clusters, debiased=False)


def test_f_pooled(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(debiased=False)

    if mod.has_constant:
        mod2 = PooledOLS(data.y, data.x)
    else:
        exog = mod.exog.dataframe.copy()
        exog['Intercept'] = 1.0
        mod2 = PooledOLS(mod.dependent.dataframe, exog)

    res2 = mod2.fit(debiased=False)

    eps = res.resids.values
    eps2 = res2.resids.values
    v1 = res.df_model - res2.df_model
    v2 = res.df_resid
    f_pool = (eps2.T @ eps2 - eps.T @ eps) / v1
    f_pool /= ((eps.T @ eps) / v2)
    f_pool = float(f_pool)
    assert_allclose(res.f_pooled.stat, f_pool)
    assert res.f_pooled.df == v1
    assert res.f_pooled.df_denom == v2

    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit(debiased=False)
    eps = res.resids.values
    eps2 = res2.resids.values
    v1 = res.df_model - res2.df_model
    v2 = res.df_resid
    f_pool = (eps2.T @ eps2 - eps.T @ eps) / v1
    f_pool /= ((eps.T @ eps) / v2)
    f_pool = float(f_pool)
    assert_allclose(res.f_pooled.stat, f_pool)
    assert res.f_pooled.df == v1
    assert res.f_pooled.df_denom == v2

    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(debiased=False)
    eps = res.resids.values
    eps2 = res2.resids.values
    v1 = res.df_model - res2.df_model
    v2 = res.df_resid
    f_pool = (eps2.T @ eps2 - eps.T @ eps) / v1
    f_pool /= ((eps.T @ eps) / v2)
    f_pool = float(f_pool)
    assert_allclose(res.f_pooled.stat, f_pool)
    assert res.f_pooled.df == v1
    assert res.f_pooled.df_denom == v2


def test_entity_other(data):
    y = PanelData(data.y)
    x = PanelData(data.x)
    c = PanelData(data.c).copy()
    missing = y.isnull | x.isnull | c.isnull
    y.drop(missing)
    x.drop(missing)
    c.drop(missing)
    c_entity = c.dataframe.copy()
    c_entity.iloc[:, 1] = y.entity_ids.squeeze()
    c_entity = c_entity.astype(np.int64)

    mod = PanelOLS(y, x, other_effects=c_entity)
    res = mod.fit(debiased=False)
    c_only = PanelData(c.dataframe.iloc[:, [0]].astype(np.int64))
    mod2 = PanelOLS(y, x, other_effects=c_only, entity_effects=True)
    res2 = mod2.fit(debiased=False)
    assert_results_equal(res, res2)


def test_other_weighted_smoke(data):
    mod = PanelOLS(data.y, data.x, weights=data.w, other_effects=data.c)
    mod.fit(debiased=False)


@pytest.mark.slow
def test_lsdv_options(data):
    mod = PanelOLS(data.y, data.x, weights=data.w)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    mod = PanelOLS(data.y, data.x, weights=data.w, entity_effects=True)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    mod = PanelOLS(data.y, data.x, time_effects=True)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    mod = PanelOLS(data.y, data.x, time_effects=True, entity_effects=True)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    c1 = PanelData(data.c).dataframe.iloc[:, [0]]
    mod = PanelOLS(data.y, data.x, entity_effects=True, other_effects=c1)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    mod = PanelOLS(data.y, data.x, time_effects=True, other_effects=c1)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    mod = PanelOLS(data.y, data.x, weights=data.w, entity_effects=True, other_effects=c1)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    mod = PanelOLS(data.y, data.x, weights=data.w, time_effects=True, other_effects=c1)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)

    mod = PanelOLS(data.y, data.x, weights=data.w, other_effects=data.c)
    res1 = mod.fit()
    res2 = mod.fit(use_lsdv=True)
    assert_results_equal(res1, res2)


def test_rsquared_inclusive_equivalence(data):
    mod = PanelOLS(data.y, data.x)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_inclusive)

    mod = PanelOLS(data.y, data.x, weights=data.w)
    res = mod.fit()
    assert_allclose(res.rsquared, res.rsquared_inclusive)


def test_panel_effects_sanity(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)

    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)

    mod = PanelOLS(data.y, data.x, weights=data.w, entity_effects=True)
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)

    mod = PanelOLS(data.y, data.x, weights=data.w, entity_effects=True, time_effects=True)
    res = mod.fit(auto_df=False, count_effects=False)
    fitted = mod.exog.values2d @ res.params.values[:, None]
    expected = fitted
    expected += res.resids.values[:, None]
    expected += res.estimated_effects.values
    assert_allclose(mod.dependent.values2d, expected)
