from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.iv.model import IV2SLS
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import PanelOLS, PooledOLS
from linearmodels.tests.panel._utility import assert_results_equal, generate_data
from linearmodels.utility import AttrDict

missing = [0.0, 0.02, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(missing, datatype, const=const, ntk=(91, 7, 5))


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
    res = mod.fit()
    res2 = IV2SLS(y, x, None, None).fit()
    assert_allclose(res.params, res2.params)


def test_const_data_only_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, weights=const_data.w)
    res = mod.fit()
    res2 = IV2SLS(y, x, None, None, weights=const_data.w).fit()
    assert_allclose(res.params, res2.params)


def test_const_data_entity(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effect=True)
    res = mod.fit()

    x = mod.exog.dataframe
    d = mod.dependent.dummies('entity', drop_first=True)
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_entity_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effect=True, weights=const_data.w)
    res = mod.fit()

    y = mod.dependent.dataframe
    w = mod.weights.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies('entity', drop_first=True)
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(y, xd, None, None, weights=w).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_time(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, time_effect=True)
    res = mod.fit()

    x = mod.exog.dataframe
    d = mod.dependent.dummies('time', drop_first=True)
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(mod.dependent.dataframe, xd, None, None).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_time_weights(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, time_effect=True, weights=const_data.w)
    res = mod.fit()

    y = mod.dependent.dataframe
    w = mod.weights.dataframe
    x = mod.exog.dataframe
    d = mod.dependent.dummies('time', drop_first=True)
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

    xd = np.c_[x.values, d.values]
    xd = pd.DataFrame(xd, index=x.index, columns=list(x.columns) + list(d.columns))

    res2 = IV2SLS(y, xd, None, None, weights=w).fit()
    assert_allclose(res.params, res2.params.iloc[:1])


def test_const_data_both(const_data):
    y, x = const_data.y, const_data.x
    mod = PanelOLS(y, x, entity_effect=True, time_effect=True)
    res = mod.fit()

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
    mod = PanelOLS(y, x, entity_effect=True, time_effect=True, weights=const_data.w)
    res = mod.fit()

    w = mod.weights.dataframe
    x = mod.exog.dataframe
    d1 = mod.dependent.dummies('entity', drop_first=True)
    d2 = mod.dependent.dummies('time', drop_first=True)
    d = np.c_[d1.values, d2.values]
    d = pd.DataFrame(d, index=x.index, columns=list(d1.columns) + list(d2.columns))
    d.iloc[:, :] = d.values - x.values @ np.linalg.lstsq(x.values, d.values)[0]

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


def test_panel_entity_lvsd(data):
    mod = PanelOLS(data.y, data.x, entity_effect=True)
    res = mod.fit()

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

    res2 = IV2SLS(y, xd, None, None).fit('unadjusted')
    assert_results_equal(res, res2, test_fit=False)


def test_panel_entity_fwl(data):
    mod = PanelOLS(data.y, data.x, entity_effect=True)
    res = mod.fit()

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

    res2 = IV2SLS(y, x, None, None).fit('unadjusted')
    assert_results_equal(res, res2, test_df=False)

#
#
# def test_panel_time(data):
#     PanelOLS(data.y, data.x, time_effect=True)
#
#
# def test_panel_entity_weighted(data):
#     PanelOLS(data.y, data.x, entity_effect=True, weights=data.w)
#
#
# def test_panel_time_weighted(data):
#     PanelOLS(data.y, data.x, time_effect=True, weights=data.w)
#
#
# def test_panel_entity_time(data):
#     PanelOLS(data.y, data.x, entity_effect=True, time_effect=True)
#
#
# def test_panel_entity_time_weighted(data):
#     PanelOLS(data.y, data.x, entity_effect=True, time_effect=True,
#              weights=data.w)
