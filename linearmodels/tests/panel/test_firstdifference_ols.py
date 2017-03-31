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
    return generate_data(missing, datatype)


def test_firstdifference_ols(data):
    mod = FirstDifferenceOLS(data.y, data.x)
    res = mod.fit(debiased=False)

    y = mod.dependent.values3d
    x = mod.exog.values3d
    dy = np.array(y[0, 1:] - y[0, :-1])
    dy = dy.T
    dy = np.reshape(dy, (dy.size, 1))

    dx = x[:, 1:] - x[:, :-1]
    _dx = []
    for dxi in dx:
        temp = dxi.T.copy()
        temp = np.reshape(temp, (temp.size, 1))
        _dx.append(temp)
    dx = np.column_stack(_dx)

    retain = np.all(np.isfinite(dy), 1) & np.all(np.isfinite(dx), 1)
    dy = dy[retain]
    dx = dx[retain]
    dy = pd.DataFrame(dy, columns=mod.dependent.dataframe.columns)
    dx = pd.DataFrame(dx, columns=mod.exog.dataframe.columns)

    res2 = IV2SLS(dy, dx, None, None).fit('unadjusted')
    assert_results_equal(res, res2)


def test_firstdifference_ols_weighted(data):
    mod = FirstDifferenceOLS(data.y, data.x, weights=data.w)
    res = mod.fit()

    y = mod.dependent.values3d
    x = mod.exog.values3d
    w = mod.weights.values3d
    dy = y[0, 1:] - y[0, :-1]
    dy = dy.T
    dy = np.reshape(dy, (dy.size, 1))

    dx = x[:, 1:] - x[:, :-1]
    _dx = []
    for dxi in dx:
        temp = dxi.T.copy()
        temp = np.reshape(temp, (temp.size, 1))
        _dx.append(temp)
    dx = np.column_stack(_dx)

    w = 1.0 / w
    sw = w[0, 1:] + w[0, :-1]
    sw = 1.0 / sw.T
    sw = np.reshape(sw, (sw.size, 1))
    sw /= np.nanmean(sw)

    retain = np.all(np.isfinite(dy), 1) & np.all(np.isfinite(dx), 1) & np.all(np.isfinite(sw), 1)
    dy = dy[retain]
    dx = dx[retain]
    sw = sw[retain]

    dy = pd.DataFrame(dy, columns=mod.dependent.dataframe.columns)
    dx = pd.DataFrame(dx, columns=mod.exog.dataframe.columns)
    sw = pd.DataFrame(sw, columns=mod.weights.dataframe.columns)

    res2 = IV2SLS(dy, dx, None, None, weights=sw).fit('unadjusted')
    assert_results_equal(res, res2)


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
