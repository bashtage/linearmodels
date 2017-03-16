import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import random_sample

from linearmodels.panel.model import PooledOLS, PanelOLS, BetweenOLS, FirstDifferenceOLS


class TestPooledOLS(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        n, t, k = 100000, 4, 5
        cls.x = random_sample((n, t, k))
        beta = np.tile(np.arange(1, k + 1), (n, t, 1))
        cls.y = (cls.x * beta).sum(axis=2) + random_sample((n, t))
        cls.y.shape = (n, t, -1)

        cls.y_pd = pd.Panel(cls.y)
        cls.x_pd = pd.Panel(cls.x)

        cls.y_xr = xr.DataArray(cls.y)
        cls.x_xr = xr.DataArray(cls.x)

        cls.mod = PooledOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        params = pols.fit()
        pols = self.mod(self.y, self.x, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        params = pols.fit()
        pols = self.mod(self.y_pd, self.x_pd, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        params = pols.fit()
        pols = self.mod(self.y_xr, self.x_xr, intercept=False)
        params = pols.fit()


class TestPanelOLS(TestPooledOLS):
    @classmethod
    def setup_class(cls):
        super(TestPanelOLS, cls).setup_class()
        cls.mod = PanelOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        params = pols.fit()

        pols = self.mod(self.y, self.x, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        params = pols.fit()

        pols = self.mod(self.y_pd, self.x_pd, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        params = pols.fit()

        pols = self.mod(self.y_xr, self.x_xr, intercept=False)
        params = pols.fit()


class TestBetweenOLS(TestPooledOLS):
    @classmethod
    def setup_class(cls):
        super(TestBetweenOLS, cls).setup_class()
        cls.mod = BetweenOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        params = pols.fit()

        pols = self.mod(self.y, self.x, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        params = pols.fit()

        pols = self.mod(self.y_pd, self.x_pd, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        params = pols.fit()

        pols = self.mod(self.y_xr, self.x_xr, intercept=False)
        params = pols.fit()


class TestFirstDifferenceOLS(TestPooledOLS):
    @classmethod
    def setup_class(cls):
        super(TestFirstDifferenceOLS, cls).setup_class()
        cls.mod = FirstDifferenceOLS

    def test_smoke(self):
        pols = self.mod(self.y, self.x)
        params = pols.fit()

        pols = self.mod(self.y, self.x, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_pd, self.x_pd)
        params = pols.fit()

        pols = self.mod(self.y_pd, self.x_pd, intercept=False)
        params = pols.fit()

        pols = self.mod(self.y_xr, self.x_xr)
        params = pols.fit()

        pols = self.mod(self.y_xr, self.x_xr, intercept=False)
        params = pols.fit()
