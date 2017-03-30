from itertools import product

import pytest

from linearmodels.iv import IV2SLS
from linearmodels.panel.model import PooledOLS
from linearmodels.tests.panel._utility import generate_data, assert_results_equal

missing = [0.0, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(missing, request.param, const=const)


def test_pooled_ols(data):
    mod = PooledOLS(data.y, data.x)
    res = mod.fit(debiased=False)

    y = mod.dependent.values2d
    x = mod.exog.values2d
    res2 = IV2SLS(y, x, None, None).fit('unadjusted')
    assert_results_equal(res, res2)


def test_pooled_ols_weighted(data):
    mod = PooledOLS(data.y, data.x, weights=data.w)
    res = mod.fit()

    y = mod.dependent.values2d
    x = mod.exog.values2d
    w = mod.weights.values2d
    res2 = IV2SLS(y, x, None, None, weights=w).fit('unadjusted')
    assert_results_equal(res, res2)