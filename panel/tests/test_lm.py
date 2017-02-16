import pandas as pd
import xarray as xr
import numpy as np
from numpy.random import random_sample
from panel.lm import PooledOLS


class TestPooledOLS(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        n, t, k = 10000, 2, 5
        cls.y = random_sample((n, t, 1))
        cls.x = random_sample((n, t, k))

        cls.y_pd = pd.Panel(cls.y)
        cls.x_pd = pd.Panel(cls.x)

        cls.y_xr = xr.DataArray(cls.y)
        cls.x_xr = xr.DataArray(cls.x)

    def test_smoke(self):
        pols = PooledOLS(self.y, self.x)
        params = pols.fit()
        print(params)
        pols = PooledOLS(self.y, self.x, intercept=False)
        params = pols.fit()
        print(params)

        pols = PooledOLS(self.y_pd, self.x_pd)
        params = pols.fit()
        print(params)
        pols = PooledOLS(self.y_pd, self.x_pd, intercept=False)
        params = pols.fit()
        print(params)

        pols = PooledOLS(self.y_xr, self.x_xr)
        params = pols.fit()
        print(params)
        pols = PooledOLS(self.y_xr, self.x_xr, intercept=False)
        params = pols.fit()
        print(params)
