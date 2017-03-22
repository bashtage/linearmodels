from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.random import random_sample
from numpy.testing import assert_allclose

from linearmodels.panel.model import PooledOLS, PanelOLS, BetweenOLS, FirstDifferenceOLS
from linearmodels.utility import AttrDict

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = ['numpy', 'pandas', 'xarray']


@pytest.fixture(params=list(product(PERC_MISSING, TYPES)),
                ids=list(map(lambda x: str(int(100 * x[0])) + '-' + str(x[1]), product(PERC_MISSING, TYPES))))
def data(request):
    missing, datatype = request.param
    n, t, k = 971, 4, 5
    x = random_sample((k, t, n))
    beta = np.arange(1, k + 1)[:, None, None]
    y = (x * beta).sum(0) + random_sample((t, n))
    if missing > 0:
        locs = np.random.choice(n * t, int(n * t * missing))
        y.flat[locs] = np.nan
        locs = np.random.choice(n * t * k, int(n * t * k * missing))
        x.flat[locs] = np.nan

    if datatype in ('pandas', 'xarray'):
        entities = ['firm' + str(i) for i in range(n)]
        time = pd.date_range('1-1-1900', periods=t, freq='A-DEC')
        vars = ['x' + str(i) for i in range(k)]
        y = pd.DataFrame(y, index=time, columns=entities)
        x = pd.Panel(x, items=vars, major_axis=time, minor_axis=entities)

    if datatype == 'xarray':
        x = xr.DataArray(x)
        y = xr.DataArray(y)

    return AttrDict(y=y, x=x)


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
