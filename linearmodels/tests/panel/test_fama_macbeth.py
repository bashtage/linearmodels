from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import FamaMacBeth
from linearmodels.tests.panel._utility import generate_data

missing = [0.0, 0.20]
datatypes = ['numpy', 'pandas', 'xarray']
has_const = [True, False]
perms = list(product(missing, datatypes, has_const))
ids = list(map(lambda s: '-'.join(map(str, s)), perms))


@pytest.fixture(params=perms, ids=ids)
def data(request):
    missing, datatype, const = request.param
    return generate_data(missing, datatype, const=const, other_effects=1, ntk=(25, 200, 5))


def test_fama_macbeth(data):
    res = FamaMacBeth(data.y, data.x).fit(debiased=True)
    y = PanelData(data.y)
    x = PanelData(data.x)
    missing = y.isnull | x.isnull
    y.drop(missing)
    x.drop(missing)
    y = y.dataframe
    x = x.dataframe
    times = y.index.levels[1]
    params = []
    for t in times:
        _y = y.xs(t, level=1)
        _x = x.xs(t, level=1)
        if _x.shape[0] < _x.shape[1]:
            continue
        _x = _x.loc[_y.index]
        params.append(np.linalg.lstsq(_x.values, _y.values)[0])
    params = np.array(params).squeeze()
    all_params = params
    params = params.mean(0)
    assert_allclose(params.squeeze(), res.params)
    e_params = all_params - params[None, :]
    ntime = e_params.shape[0]
    cov = e_params.T @ e_params / ntime / (ntime - 1)
    assert_allclose(cov, res.cov.values)

    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()


def test_unknown_cov_type(data):
    with pytest.raises(ValueError):
        FamaMacBeth(data.y, data.x).fit(cov_type='unknown')


def test_fama_macbeth_kernel_smoke(data):
    FamaMacBeth(data.y, data.x).fit(cov_type='kernel')
    FamaMacBeth(data.y, data.x).fit(cov_type='kernel', kernel='bartlett')
    FamaMacBeth(data.y, data.x).fit(cov_type='kernel', kernel='newey-west')
    FamaMacBeth(data.y, data.x).fit(cov_type='kernel', kernel='parzen')
    FamaMacBeth(data.y, data.x).fit(cov_type='kernel', kernel='qs')
    FamaMacBeth(data.y, data.x).fit(cov_type='kernel', bandwidth=3)
    res = FamaMacBeth(data.y, data.x).fit(cov_type='kernel', kernel='andrews')

    d = dir(res)
    for key in d:
        if not key.startswith('_'):
            val = getattr(res, key)
            if callable(val):
                val()
