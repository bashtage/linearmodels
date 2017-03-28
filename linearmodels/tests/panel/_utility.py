import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import random_sample

from linearmodels.utility import AttrDict


def lvsd(y: pd.DataFrame, x: pd.DataFrame, w=None, has_const=False, entity=False, time=False,
         general=None):
    x_orig = x
    nvar = x.shape[1]
    temp = x.reset_index()
    cat_index = temp.index
    if entity:
        cat = pd.Categorical(temp.iloc[:, 0])
        cat.index = cat_index
        dummies = pd.get_dummies(cat, drop_first=has_const)
        x = pd.DataFrame(np.c_[x.values, dummies.values.astype(np.float64)],
                         index=x.index,
                         columns=list(x.columns) + list(dummies.columns))
    if time:
        cat = pd.Categorical(temp.iloc[:, 1])
        cat.index = cat_index
        dummies = pd.get_dummies(cat, drop_first=(has_const or entity))
        x = pd.DataFrame(np.c_[x.values, dummies.values.astype(np.float64)],
                         index=x.index,
                         columns=list(x.columns) + list(dummies.columns))
    if general is not None:
        cat = pd.Categorical(general)
        cat.index = cat_index
        dummies = pd.get_dummies(cat, drop_first=(has_const or entity or time))
        x = pd.DataFrame(np.c_[x.values, dummies.values.astype(np.float64)],
                         index=x.index,
                         columns=list(x.columns) + list(dummies.columns))
    if w is None:
        w = np.ones_like(y)

    wy = w * y.values
    wx = w * x.values
    params = np.linalg.lstsq(wx,wy)[0]
    params = params.squeeze()

    return params[:nvar]


def generate_data(missing, datatype):
    np.random.seed(12345)

    n, t, k = 971, 7, 5
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
