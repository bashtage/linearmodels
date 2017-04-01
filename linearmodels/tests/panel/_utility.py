import numpy as np
import pandas as pd
import xarray as xr
from numpy.random import standard_normal
from numpy.testing import assert_allclose
from pandas.util.testing import assert_frame_equal, assert_series_equal

from linearmodels.utility import AttrDict


def lvsd(y: pd.DataFrame, x: pd.DataFrame, w=None, has_const=False, entity=False, time=False,
         general=None):
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
    params = np.linalg.lstsq(wx, wy)[0]
    params = params.squeeze()

    return params[:nvar]


def generate_data(missing, datatype, const=False, ntk=(971, 7, 5)):
    np.random.seed(12345)

    n, t, k = ntk
    k += const
    x = standard_normal((k, t, n))
    beta = np.arange(1, k + 1)[:, None, None] / k
    y = (x * beta).sum(0) + standard_normal((t, n)) + 2 * standard_normal((t, 1))
    w = np.random.chisquare(5, (t, n)) / 5
    if const:
        x[0] = 1.0

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
        w = pd.DataFrame(w, index=time, columns=entities)
        x = pd.Panel(x, items=vars, major_axis=time, minor_axis=entities)

    if datatype == 'xarray':
        x = xr.DataArray(x)
        y = xr.DataArray(y)
        w = xr.DataArray(w)

    return AttrDict(y=y, x=x, w=w)


def assert_results_equal(res1, res2, n=None, test_fit=True, test_df=True):
    if n is None:
        n = min(res1.params.shape[0], res2.params.shape[0])

    assert_series_equal(res1.params.iloc[:n], res2.params.iloc[:n])
    assert_series_equal(res1.pvalues.iloc[:n], res2.pvalues.iloc[:n])
    assert_series_equal(res1.tstats.iloc[:n], res2.tstats.iloc[:n])
    assert_frame_equal(res1.cov.iloc[:n, :n], res2.cov.iloc[:n, :n])
    assert_frame_equal(res1.conf_int().iloc[:n], res2.conf_int().iloc[:n])
    assert_allclose(res1.s2, res2.s2)
    if test_df:
        assert_allclose(res1.df_model, res2.df_model)
        assert_allclose(res1.df_resid, res2.df_resid)
    if test_fit:
        assert_allclose(res1.rsquared, res2.rsquared)
        assert_allclose(res1.total_ss, res2.total_ss)
        assert_allclose(res1.resid_ss, res2.resid_ss)
        assert_allclose(res1.model_ss, res2.model_ss)
