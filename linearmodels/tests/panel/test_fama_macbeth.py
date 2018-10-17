from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from linearmodels.compat.numpy import lstsq
from linearmodels.panel.data import PanelData
from linearmodels.panel.model import FamaMacBeth
from linearmodels.tests.panel._utility import generate_data, datatypes, assert_frame_similar
from linearmodels.utility import MissingValueWarning, InferenceUnavailableWarning

pytestmark = pytest.mark.filterwarnings('ignore::linearmodels.utility.MissingValueWarning')

missing = [0.0, 0.20]
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
        params.append(lstsq(_x.values, _y.values)[0])
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


def test_fitted_effects_residuals(data):
    mod = FamaMacBeth(data.y, data.x)
    res = mod.fit()

    expected = mod.exog.values2d @ res.params.values
    expected = pd.DataFrame(expected, index=mod.exog.index, columns=['fitted_values'])
    assert_allclose(res.fitted_values, expected)
    assert_frame_similar(res.fitted_values, expected)

    expected.iloc[:, 0] = mod.dependent.values2d - expected.values
    expected.columns = ['idiosyncratic']
    assert_allclose(res.idiosyncratic, expected)
    assert_frame_similar(res.idiosyncratic, expected)

    expected.iloc[:, 0] = np.nan
    expected.columns = ['estimated_effects']
    assert_allclose(res.estimated_effects, expected)
    assert_frame_similar(res.estimated_effects, expected)


@pytest.mark.filterwarnings('always::linearmodels.utility.MissingValueWarning')
def test_block_size_warnings():
    y = np.arange(12.0)[:, None]
    x = np.ones((12, 3))
    x[:, 1] = np.arange(12.0)
    x[:, 2] = np.arange(12.0) ** 2
    idx = pd.MultiIndex.from_product([['a', 'b', 'c'], pd.date_range('2000-1-1', periods=4)])
    y = pd.DataFrame(y, index=idx, columns=['y'])
    x = pd.DataFrame(x, index=idx, columns=['x1', 'x2', 'x3'])
    with pytest.warns(MissingValueWarning):
        FamaMacBeth(y.iloc[:11], x.iloc[:11])
    with pytest.warns(InferenceUnavailableWarning):
        FamaMacBeth(y.iloc[::4], x.iloc[::4])


def test_block_size_error():
    y = np.arange(12.0)[:, None]
    x = np.ones((12, 2))
    x[1::4, 1] = 2
    x[2::4, 1] = 3
    idx = pd.MultiIndex.from_product([['a', 'b', 'c'], pd.date_range('2000-1-1', periods=4)])
    y = pd.DataFrame(y, index=idx, columns=['y'])
    x = pd.DataFrame(x, index=idx, columns=['x1', 'x2'])
    with pytest.raises(ValueError):
        FamaMacBeth(y, x)
