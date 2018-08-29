import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_allclose
from pandas import DataFrame, Categorical, get_dummies, date_range
from pandas.testing import assert_frame_equal, assert_series_equal

from linearmodels.compat.numpy import lstsq
from linearmodels.panel.data import PanelData
from linearmodels.utility import panel_to_frame

try:
    import xarray  # flake8: noqa

    MISSING_XARRAY = False
except ImportError:
    MISSING_XARRAY = True
from linearmodels.utility import AttrDict

datatypes = ['numpy', 'pandas']
if not MISSING_XARRAY:
    datatypes += ['xarray']


def lsdv(y: DataFrame, x: DataFrame, has_const=False, entity=False, time=False,
         general=None):
    nvar = x.shape[1]
    temp = x.reset_index()
    cat_index = temp.index
    if entity:
        cat = Categorical(temp.iloc[:, 0])
        cat.index = cat_index
        dummies = get_dummies(cat, drop_first=has_const)
        x = DataFrame(np.c_[x.values, dummies.values.astype(np.float64)],
                         index=x.index,
                         columns=list(x.columns) + list(dummies.columns))
    if time:
        cat = Categorical(temp.iloc[:, 1])
        cat.index = cat_index
        dummies = get_dummies(cat, drop_first=(has_const or entity))
        x = DataFrame(np.c_[x.values, dummies.values.astype(np.float64)],
                         index=x.index,
                         columns=list(x.columns) + list(dummies.columns))
    if general is not None:
        cat = Categorical(general)
        cat.index = cat_index
        dummies = get_dummies(cat, drop_first=(has_const or entity or time))
        x = DataFrame(np.c_[x.values, dummies.values.astype(np.float64)],
                         index=x.index,
                         columns=list(x.columns) + list(dummies.columns))
    w = np.ones_like(y)

    wy = w * y.values
    wx = w * x.values
    params = lstsq(wx, wy)[0]
    params = params.squeeze()

    return params[:nvar]


def generate_data(missing, datatype, const=False, ntk=(971, 7, 5), other_effects=0, rng=None):
    if rng is None:
        np.random.seed(12345)
    else:
        np.random.set_state(rng.get_state())

    n, t, k = ntk
    k += const
    x = standard_normal((k, t, n))
    beta = np.arange(1, k + 1)[:, None, None] / k
    y = (x * beta).sum(0) + standard_normal((t, n)) + 2 * standard_normal((1, n))
    w = np.random.chisquare(5, (t, n)) / 5
    c = None
    if other_effects == 1:
        cats = ['Industries']
    else:
        cats = ['cat.' + str(i) for i in range(other_effects)]
    if other_effects:
        c = np.random.randint(0, 4, (other_effects, t, n))

    vcats = ['varcat.' + str(i) for i in range(2)]
    vc2 = np.ones((2, t, 1)) @ np.random.randint(0, n // 2, (2, 1, n))
    vc1 = vc2[[0]]

    if const:
        x[0] = 1.0

    if missing > 0:
        locs = np.random.choice(n * t, int(n * t * missing))
        y.flat[locs] = np.nan
        locs = np.random.choice(n * t * k, int(n * t * k * missing))
        x.flat[locs] = np.nan

    if datatype in ('pandas', 'xarray'):
        entities = ['firm' + str(i) for i in range(n)]
        time = date_range('1-1-1900', periods=t, freq='A-DEC')
        var_names = ['x' + str(i) for i in range(k)]
        # y = DataFrame(y, index=time, columns=entities)
        y = panel_to_frame(y[None], items=['y'], major_axis=time, minor_axis=entities, swap=True)
        w = panel_to_frame(w[None], items=['w'], major_axis=time, minor_axis=entities, swap=True)
        w = w.reindex(y.index)
        x = panel_to_frame(x, items=var_names, major_axis=time, minor_axis=entities, swap=True)
        x = x.reindex(y.index)
        c = panel_to_frame(c, items=cats, major_axis=time, minor_axis=entities, swap=True)
        c = c.reindex(y.index)
        vc1 = panel_to_frame(vc1, items=vcats[:1], major_axis=time, minor_axis=entities, swap=True)
        vc1 = vc1.reindex(y.index)
        vc2 = panel_to_frame(vc2, items=vcats, major_axis=time, minor_axis=entities, swap=True)
        vc2 = vc2.reindex(y.index)

    if datatype == 'xarray':
        # TODO: This is broken now, need to transfor multiindex to xarray 3d
        import xarray as xr
        x = xr.DataArray(PanelData(x).values3d,
                         coords={'entities': entities, 'time': time,
                                 'vars': var_names},
                         dims=['vars', 'time', 'entities'])
        y = xr.DataArray(PanelData(y).values3d,
                         coords={'entities': entities, 'time': time,
                                 'vars': ['y']},
                         dims=['vars', 'time', 'entities'])
        w = xr.DataArray(PanelData(w).values3d,
                         coords={'entities': entities, 'time': time,
                                 'vars': ['w']},
                         dims=['vars', 'time', 'entities'])
        if c.shape[1] > 0:
            c = xr.DataArray(PanelData(c).values3d,
                             coords={'entities': entities, 'time': time,
                                     'vars': c.columns},
                             dims=['vars', 'time', 'entities'])
        vc1 = xr.DataArray(PanelData(vc1).values3d,
                           coords={'entities': entities, 'time': time,
                                   'vars': vc1.columns},
                           dims=['vars', 'time', 'entities'])
        vc2 = xr.DataArray(PanelData(vc2).values3d,
                           coords={'entities': entities, 'time': time,
                                   'vars': vc2.columns},
                           dims=['vars', 'time', 'entities'])

    if rng is not None:
        rng.set_state(np.random.get_state())

    return AttrDict(y=y, x=x, w=w, c=c, vc1=vc1, vc2=vc2)


def assert_results_equal(res1, res2, test_fit=True, test_df=True):
    n = min(res1.params.shape[0], res2.params.shape[0])

    assert_series_equal(res1.params.iloc[:n], res2.params.iloc[:n])
    assert_series_equal(res1.pvalues.iloc[:n], res2.pvalues.iloc[:n])
    assert_series_equal(res1.tstats.iloc[:n], res2.tstats.iloc[:n])
    assert_frame_equal(res1.cov.iloc[:n, :n], res2.cov.iloc[:n, :n])
    assert_frame_equal(res1.conf_int().iloc[:n], res2.conf_int().iloc[:n])
    assert_allclose(res1.s2, res2.s2)

    delta = 1 + (res1.resids.values - res2.resids.values) / max(
        res1.resids.std(),
        res2.resids.std())
    assert_allclose(delta, np.ones_like(delta))
    delta = 1 + (res1.wresids.values - res2.wresids.values) / max(
        res1.wresids.std(),
        res2.wresids.std())
    assert_allclose(delta, np.ones_like(delta))

    if test_df:
        assert_allclose(res1.df_model, res2.df_model)
        assert_allclose(res1.df_resid, res2.df_resid)

    if test_fit:
        assert_allclose(res1.rsquared, res2.rsquared)
        assert_allclose(res1.total_ss, res2.total_ss)
        assert_allclose(res1.resid_ss, res2.resid_ss)
        assert_allclose(res1.model_ss, res2.model_ss)


def assert_frame_similar(result, expected):
    r = result.copy()
    r.iloc[:, :] = 0.0
    e = expected.copy()
    e.iloc[:, :] = 0.0
    assert_frame_equal(r, e)
