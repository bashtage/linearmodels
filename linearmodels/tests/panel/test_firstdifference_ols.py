from itertools import product

import numpy as np
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS
from linearmodels.panel.model import FirstDifferenceOLS
from linearmodels.tests.panel._utility import assert_results_equal, generate_data

missing = [0.0, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
perms = list(product(missing, datatypes))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype = request.param
    return generate_data(missing, datatype, other_effects=1)


def test_firstdifference_ols(data):
    mod = FirstDifferenceOLS(data.y, data.x)
    res = mod.fit(debiased=False)

    y = mod.dependent.values3d
    x = mod.exog.values3d
    dy = np.array(y[0, 1:] - y[0, :-1])
    dy = pd.DataFrame(dy, index=mod.dependent.panel.major_axis[1:],
                      columns=mod.dependent.panel.minor_axis)
    dy = dy.T.stack()
    dy = dy.reindex(mod.dependent.dataframe.index)

    dx = x[:, 1:] - x[:, :-1]
    _dx = {}
    for i, dxi in enumerate(dx):
        temp = pd.DataFrame(dxi, index=mod.dependent.panel.major_axis[1:],
                            columns=mod.dependent.panel.minor_axis)
        temp = temp.T.stack()
        temp = temp.reindex(mod.dependent.dataframe.index)
        _dx[mod.exog.vars[i]] = temp
    dx = pd.DataFrame(index=_dx[mod.exog.vars[i]].index)
    for key in _dx:
        dx[key] = _dx[key]
    dx = dx[mod.exog.vars]
    drop = dy.isnull() | np.any(dx.isnull(), 1)
    dy = dy.loc[~drop]
    dx = dx.loc[~drop]

    ols_mod = IV2SLS(dy, dx, None, None)
    ols_res = ols_mod.fit('unadjusted')
    assert_results_equal(res, ols_res)

    res = mod.fit(cov_type='robust')
    ols_res = ols_mod.fit('robust')
    assert_results_equal(res, ols_res)

    clusters = mod.dependent.dataframe.copy()
    for entity in mod.dependent.entities:
        clusters.loc[entity] = np.random.randint(9)

    ols_clusters = clusters.copy()
    index = mod.dependent.first_difference().dataframe.index
    ols_clusters = ols_clusters.loc[index]

    res = mod.fit(cov_type='clustered', clusters=clusters)
    ols_res = ols_mod.fit(cov_type='clustered', clusters=ols_clusters)
    assert_results_equal(res, ols_res)

    entity_clusters = mod.dependent.first_difference().entity_ids
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    ols_res = ols_mod.fit(cov_type='clustered', clusters=entity_clusters)
    assert_results_equal(res, ols_res)

    ols_clusters['entity.clusters'] = entity_clusters
    ols_clusters = ols_clusters.astype(np.int32)
    res = mod.fit(cov_type='clustered', cluster_entity=True, clusters=clusters)
    ols_res = ols_mod.fit(cov_type='clustered', clusters=ols_clusters)
    assert_results_equal(res, ols_res)


def test_firstdifference_ols_weighted(data):
    mod = FirstDifferenceOLS(data.y, data.x, weights=data.w)
    res = mod.fit()

    y = mod.dependent.values3d
    x = mod.exog.values3d
    dy = np.array(y[0, 1:] - y[0, :-1])
    dy = pd.DataFrame(dy, index=mod.dependent.panel.major_axis[1:],
                      columns=mod.dependent.panel.minor_axis)
    dy = dy.T.stack()
    dy = dy.reindex(mod.dependent.dataframe.index)

    dx = x[:, 1:] - x[:, :-1]
    _dx = {}
    for i, dxi in enumerate(dx):
        temp = pd.DataFrame(dxi, index=mod.dependent.panel.major_axis[1:],
                            columns=mod.dependent.panel.minor_axis)
        temp = temp.T.stack()
        temp = temp.reindex(mod.dependent.dataframe.index)
        _dx[mod.exog.vars[i]] = temp
    dx = pd.DataFrame(index=_dx[mod.exog.vars[i]].index)
    for key in _dx:
        dx[key] = _dx[key]
    dx = dx[mod.exog.vars]

    w = mod.weights.values3d
    w = 1.0 / w
    sw = w[0, 1:] + w[0, :-1]
    sw = pd.DataFrame(sw, index=mod.dependent.panel.major_axis[1:],
                      columns=mod.dependent.panel.minor_axis)
    sw = sw.T.stack()
    sw = sw.reindex(mod.dependent.dataframe.index)
    sw = 1.0 / sw
    sw = sw / sw.mean()

    drop = dy.isnull() | np.any(dx.isnull(), 1) | sw.isnull()
    dy = dy.loc[~drop]
    dx = dx.loc[~drop]
    sw = sw.loc[~drop]

    ols_mod = IV2SLS(dy, dx, None, None, weights=sw)
    ols_res = ols_mod.fit('unadjusted')
    assert_results_equal(res, ols_res)

    res = mod.fit(cov_type='robust')
    ols_res = ols_mod.fit('robust')
    assert_results_equal(res, ols_res)

    clusters = mod.dependent.dataframe.copy()
    for entity in mod.dependent.entities:
        clusters.loc[entity] = np.random.randint(9)

    ols_clusters = clusters.copy()
    index = mod.dependent.first_difference().dataframe.index
    ols_clusters = ols_clusters.loc[index]

    res = mod.fit(cov_type='clustered', clusters=clusters)
    ols_res = ols_mod.fit(cov_type='clustered', clusters=ols_clusters)
    assert_results_equal(res, ols_res)


def test_first_difference_errors(data):
    if isinstance(data.x, pd.Panel):
        x = data.x.iloc[:, [0], :]
        y = data.y.iloc[[0], :]
    else:
        x = data.x[:, [0], :]
        y = data.y[[0], :]
    with pytest.raises(ValueError):
        FirstDifferenceOLS(y, x)

    if not isinstance(data.x, pd.Panel):
        return
    x = data.x.copy()
    x['Intercept'] = 1.0
    with pytest.raises(ValueError):
        FirstDifferenceOLS(data.y, x)


def test_results_access(data):
    mod = FirstDifferenceOLS(data.y, data.x)
    res = mod.fit()
    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()

def test_firstdifference_error(data):
    mod = FirstDifferenceOLS(data.y, data.x)

    clusters = mod.dependent.dataframe.copy()
    for entity in mod.dependent.entities:
        clusters.loc[entity] = np.random.randint(9)
    clusters.iloc[::3,:] = clusters.iloc[::3,:] + 1

    with pytest.raises(ValueError):
        mod.fit(cov_type='clustered', clusters=clusters)
