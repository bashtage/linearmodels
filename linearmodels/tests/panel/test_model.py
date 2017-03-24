from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.random import random_sample
from numpy.testing import assert_allclose

from linearmodels.panel.model import PooledOLS, PanelOLS, BetweenOLS, FirstDifferenceOLS, \
    AmbiguityError
from linearmodels.tests.panel._utility import lvsd, generate_data

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = ['numpy', 'pandas', 'xarray']


@pytest.fixture(params=list(product(PERC_MISSING, TYPES)),
                ids=list(map(lambda x: str(int(100 * x[0])) + '-' + str(x[1]),
                             product(PERC_MISSING, TYPES))))
def data(request):
    missing, datatype = request.param
    return generate_data(missing, datatype)


def test_pooled_ols(data):
    PooledOLS(data.y, data.x).fit()


def test_between(data):
    BetweenOLS(data.y, data.x).fit()


def test_first_differnce(data):
    FirstDifferenceOLS(data.y, data.x).fit()


def test_panel_ols(data):
    PanelOLS(data.y, data.x).fit()
    PanelOLS(data.y, data.x, entity_effect=True).fit()
    PanelOLS(data.y, data.x, time_effect=True).fit()


def test_pooled_ols_formula(data):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    formula = 'y ~ x1 + x2'
    mod = PooledOLS.from_formula(formula, joined)
    res = mod.fit()
    res2 = PooledOLS(joined['y'], joined[['x1', 'x2']]).fit()
    assert_allclose(res, res2)
    assert mod.formula == formula


def test_panel_ols_formula(data):
    if not isinstance(data.y, pd.DataFrame):
        return
    joined = data.x
    joined['y'] = data.y
    formula = 'y ~ x1 + x2'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula

    formula = 'y ~ x1 + x2 + EntityEffect'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.entity_effect is True
    assert mod.time_effect is False

    formula = 'y ~ x1 + x2 + TimeEffect'
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.time_effect is True
    assert mod.entity_effect is False

    formula = 'y ~ x1 + EntityEffect + TimeEffect + x2 '
    mod = PanelOLS.from_formula(formula, joined)
    assert mod.formula == formula
    assert mod.entity_effect is True
    assert mod.time_effect is True

    formula = 'y ~ x1 + EntityEffect + FixedEffect + x2 '
    with pytest.raises(ValueError):
        PanelOLS.from_formula(formula, joined)


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


def test_weights(data):
    n = np.prod(data.y.shape)
    weights = 1 + np.random.random_sample(n)
    PooledOLS(data.y, data.x, weights=weights).fit()

    n = data.y.shape[0]
    weights = 1 + np.random.random_sample(n)
    PooledOLS(data.y, data.x, weights=weights).fit()

    n = data.y.shape[1]
    weights = 1 + np.random.random_sample(n)
    PooledOLS(data.y, data.x, weights=weights).fit()

    weights = 1 + np.random.random_sample(data.y.shape)
    PooledOLS(data.y, data.x, weights=weights).fit()


def test_weight_ambiguity(data):
    t = data.y.shape[0]
    if isinstance(data.x, pd.Panel):
        x = data.x.iloc[:, :, :t]
    else:
        x = data.x[:, :, :t]
    y = data.y
    weights = 1 + np.random.random_sample(t)
    with pytest.raises(AmbiguityError):
        PooledOLS(y, x, weights=weights)


def test_weight_incorrect_shape(data):
    weights = np.ones(np.prod(data.y.shape) - 1)
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=weights)

    weights = np.ones((data.y.shape[0], data.y.shape[1] - 1))
    with pytest.raises(ValueError):
        PanelOLS(data.y, data.x, weights=weights)


def test_panel_lvsd(data):
    mod = PanelOLS(data.y, data.x, entity_effect=True)
    y, x = mod.dependent.dataframe, mod.exog.dataframe
    res = mod.fit()
    expected = lvsd(y, x, has_const=False, entity=True)
    assert_allclose(res.squeeze(), expected)

    mod = PanelOLS(data.y, data.x, time_effect=True)
    res = mod.fit()
    expected = lvsd(y, x, has_const=False, time=True)
    assert_allclose(res.squeeze(), expected)

    mod = PanelOLS(data.y, data.x, entity_effect=True, time_effect=True)
    res = mod.fit()
    expected = lvsd(y, x, has_const=False, entity=True, time=True)
    assert_allclose(res.squeeze(), expected, rtol=1e-4)


class TestPooledOLS(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        n, t, k = 10000, 4, 5
        cls.x = random_sample((k, t, n))
        beta = np.arange(1, k + 1)[:, None, None]
        cls.y = (cls.x * beta).sum(0) + random_sample((t, n))

        cls.y_pd = pd.DataFrame(cls.y)
        cls.x_pd = pd.Panel(cls.x)

        cls.y_xr = xr.DataArray(cls.y)
        cls.x_xr = xr.DataArray(cls.x)

        cls.mod = PooledOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        pols.fit()
        pols = self.mod(self.y, self.x)
        pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()
        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()
        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()


class TestPanelOLS(TestPooledOLS):
    @classmethod
    def setup_class(cls):
        super(TestPanelOLS, cls).setup_class()
        cls.mod = PanelOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        pols.fit()

        pols = self.mod(self.y, self.x)
        pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()


class TestBetweenOLS(TestPooledOLS):
    @classmethod
    def setup_class(cls):
        super(TestBetweenOLS, cls).setup_class()
        cls.mod = BetweenOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        pols.fit()

        pols = self.mod(self.y, self.x)
        pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()


class TestFirstDifferenceOLS(TestPooledOLS):
    @classmethod
    def setup_class(cls):
        super(TestFirstDifferenceOLS, cls).setup_class()
        cls.mod = FirstDifferenceOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        pols.fit()

        pols = self.mod(self.y, self.x)
        pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        pols.fit()
