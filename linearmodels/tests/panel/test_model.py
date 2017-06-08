from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_equal

from linearmodels.panel.data import PanelData
from linearmodels.panel.model import (AbsorbingEffectError, AmbiguityError,
                                      PanelOLS)
from linearmodels.tests.panel._utility import generate_data, lsdv

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = ['numpy', 'pandas', 'xarray']


@pytest.fixture(params=list(product(PERC_MISSING, TYPES)),
                ids=list(map(lambda x: str(int(100 * x[0])) + '-' + str(x[1]),
                             product(PERC_MISSING, TYPES))))
def data(request):
    missing, datatype = request.param
    rng = np.random.RandomState(12345)
    return generate_data(missing, datatype, ntk=(131, 4, 3), rng=rng)


def test_panel_ols(data):
    PanelOLS(data.y, data.x).fit()
    PanelOLS(data.y, data.x, entity_effects=True).fit()
    PanelOLS(data.y, data.x, time_effects=True).fit()


def test_valid_weight_shape(data):
    # Same size
    n = np.prod(data.y.shape)
    weights = 1 + np.random.random_sample(n)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    missing = PanelData(data.y).isnull | PanelData(data.x).isnull
    expected = weights[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert_equal(w, expected)

    # Per time
    n = data.y.shape[0]
    weights = 1 + np.random.random_sample(n)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    expected = weights[:, None] @ np.ones((1, data.y.shape[1]))
    expected = expected.T.ravel()
    expected = expected[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert_equal(w, expected)

    # Per entity
    n = data.y.shape[1]
    weights = 1 + np.random.random_sample(n)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    expected = np.ones((data.y.shape[0], 1)) @ weights[None, :]
    expected = expected.T.ravel()
    expected = expected[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert_equal(w, expected)

    weights = 1 + np.random.random_sample(data.y.shape)
    mod = PanelOLS(data.y, data.x, weights=weights)
    mod.fit()
    w = mod.weights.values2d
    expected = weights.T.ravel()
    expected = expected[~missing.squeeze()][:, None]
    expected = expected / expected.mean()
    assert_equal(w, expected)


def test_weight_incorrect_shape(data):
    weights = np.ones(np.prod(data.y.shape) - 1)
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=weights)

    weights = np.ones((data.y.shape[0], data.y.shape[1] - 1))
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=weights)


def test_invalid_weight_values(data):
    w = PanelData(data.w)
    w.dataframe.iloc[::13, :] = 0.0
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=w)

    w = PanelData(data.w)
    w.dataframe.iloc[::13, :] = -0.0
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=w)

    w = PanelData(data.w)
    w.dataframe.iloc[::29, :] = -1.0
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=w)


def test_panel_lsdv(data):
    mod = PanelOLS(data.y, data.x, entity_effects=True)
    y, x = mod.dependent.dataframe, mod.exog.dataframe
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, entity=True)
    assert_allclose(res.params.squeeze(), expected)

    mod = PanelOLS(data.y, data.x, time_effects=True)
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, time=True)
    assert_allclose(res.params.squeeze(), expected)

    mod = PanelOLS(data.y, data.x, entity_effects=True, time_effects=True)
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, entity=True, time=True)
    assert_allclose(res.params.squeeze(), expected, rtol=1e-4)

    other = y.copy()
    other.iloc[:, :] = 0
    other = other.astype(np.int64)
    skip = other.shape[0] // 3
    for i in range(skip):
        other.iloc[i::skip] = i

    mod = PanelOLS(y, x, other_effects=other)
    res = mod.fit()
    expected = lsdv(y, x, has_const=False, general=other.iloc[:, 0].values)
    assert_allclose(res.params.squeeze(), expected, rtol=1e-4)


def test_incorrect_weight_shape(data):
    if isinstance(data.w, xr.DataArray):
        return

    w = data.w
    if isinstance(w, pd.DataFrame):
        w = w.iloc[:3]
        w = pd.Panel({'weights': w})
    else:
        w = w[:3]
        w = w[None, :, :]

    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=w)


def test_weight_ambiguity(data):
    t = data.y.shape[0]
    if isinstance(data.x, pd.Panel):
        x = data.x.iloc[:, :, :t]
    else:
        x = data.x[:, :, :t]
    y = data.y
    weights = 1 + np.random.random_sample(t)
    with pytest.raises(AmbiguityError):
        PanelOLS(y, x, weights=weights)


def test_absorbing_effect(data):
    if not isinstance(data.x, pd.Panel):
        return
    x = data.x.copy()
    temp = data.x.iloc[0].copy()
    temp.values[:, :] = 1.0
    n = temp.shape[1]
    temp.values[:, n // 2:] = 0
    x['Intercept'] = 1.0
    x['absorbed'] = temp
    with pytest.raises(AbsorbingEffectError):
        PanelOLS(data.y, x, entity_effects=True).fit()


def test_all_missing(data):
    y = PanelData(data.y)
    x = PanelData(data.x)
    missing = y.isnull | x.isnull
    y.drop(missing)
    x.drop(missing)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        PanelOLS(y.dataframe, x.dataframe).fit()
        assert len(w) == 0
