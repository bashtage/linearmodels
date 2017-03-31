from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linearmodels.iv import IV2SLS
from linearmodels.panel.model import PooledOLS
from linearmodels.tests.panel._utility import assert_results_equal, generate_data

missing = [0.0, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(missing, datatype, const=const)


def test_pooled_ols(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit(debiased=False)

    y = mod.dependent.dataframe.copy()
    x = mod.exog.dataframe.copy()
    y.index = np.arange(len(y))
    x.index = y.index

    res2 = IV2SLS(y, x, None, None).fit('unadjusted')
    assert_results_equal(res, res2)


def test_pooled_ols_weighted(data):
    mod = PooledOLS(data.y, data.x, weights=data.w)
    res = mod.fit()

    y = mod.dependent.dataframe
    x = mod.exog.dataframe
    w = mod.weights.dataframe
    y.index = np.arange(len(y))
    w.index = x.index = y.index

    res2 = IV2SLS(y, x, None, None, weights=w).fit('unadjusted')
    assert_results_equal(res, res2)


def test_diff_data_size(data):
    if isinstance(data.x, pd.Panel):
        x = data.x.iloc[:, :, :-1]
        y = data.y
    elif isinstance(data.x, xr.DataArray):
        x = data.x[:, :-1]
        y = data.y[:, :-1]
    else:
        x = data.x
        y = data.y[:-1]
    with pytest.raises(ValueError):
        PooledOLS(y, x)


def test_rank_deficient_array(data):
    x = data.x
    if isinstance(data.x, pd.Panel):
        x.iloc[1] = x.iloc[0]
    else:
        x[1] = x[0]
    with pytest.raises(ValueError):
        PooledOLS(data.y, x)
