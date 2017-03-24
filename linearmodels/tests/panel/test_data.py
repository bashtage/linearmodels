from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.linalg import pinv
from numpy.testing import assert_equal, assert_allclose
from pandas.util.testing import assert_frame_equal, assert_panel_equal

from linearmodels.panel.data import PanelData
from linearmodels.tests.panel._utility import generate_data

PERC_MISSING = [0, 0.02, 0.10, 0.33]
TYPES = ['numpy', 'pandas', 'xarray']


@pytest.fixture(params=list(product(PERC_MISSING, TYPES)),
                ids=list(map(lambda x: str(int(100 * x[0])) + '-' + str(x[1]),
                             product(PERC_MISSING, TYPES))))
def data(request):
    missing, datatype = request.param
    return generate_data(missing, datatype)


@pytest.fixture
def panel():
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    major = pd.date_range('12-31-1999', periods=7)
    items = ['var.{0}'.format(i) for i in range(1, k + 1)]
    minor = ['entities.{0}'.format(i) for i in range(1, n + 1)]
    return pd.Panel(x, items=items, major_axis=major, minor_axis=minor)


def test_numpy_3d():
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    dh = PanelData(x)
    assert_equal(np.reshape(x, (k, t, n)), dh.a3d)
    assert dh.nentity == n
    assert dh.nobs == t
    assert dh.nvar == k
    assert_equal(np.reshape(x.T, (n * t, k)), dh.a2d)
    items = ['entity.{0}'.format(i) for i in range(n)]
    obs = [i for i in range(t)]
    vars = ['x.{0}'.format(i) for i in range(k)]
    expected = pd.Panel(np.reshape(x, (k, t, n)), items=vars,
                        major_axis=obs, minor_axis=items)
    assert_panel_equal(dh.panel, expected)
    expected_frame = expected.swapaxes(1, 2).to_frame()
    expected_frame.index.levels[0].name = 'entity'
    expected_frame.index.levels[1].name = 'time'
    assert_frame_equal(dh.dataframe, expected_frame)


def test_numpy_1d():
    n, t, k = 11, 1, 1
    x = np.random.random(n)
    with pytest.raises(ValueError):
        PanelData(x)


def test_numpy_2d():
    n, t, k = 11, 7, 1
    x = np.random.random((t, n))
    dh = PanelData(x)
    assert_equal(np.reshape(x, (k, t, n)), dh.a3d)
    assert dh.nentity == n
    assert dh.nobs == t
    assert dh.nvar == k
    assert_equal(np.reshape(x.T, (n * t, k)), dh.a2d)
    items = ['entity.{0}'.format(i) for i in range(n)]
    obs = [i for i in range(t)]
    vars = ['x.{0}'.format(i) for i in range(k)]
    expected = pd.Panel(np.reshape(x, (k, t, n)), items=vars,
                        major_axis=obs, minor_axis=items)
    assert_panel_equal(dh.panel, expected)
    expected_frame = expected.swapaxes(1, 2).to_frame()
    expected_frame.index.levels[0].name = 'entity'
    expected_frame.index.levels[1].name = 'time'
    assert_frame_equal(dh.dataframe, expected_frame)


def test_pandas_panel():
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    major = pd.date_range('12-31-1999', periods=7)
    items = ['var.{0}'.format(i) for i in range(1, k + 1)]
    minor = ['entities.{0}'.format(i) for i in range(1, n + 1)]
    x = pd.Panel(x, items=items, major_axis=major, minor_axis=minor)
    dh = PanelData(x)
    assert dh.nentity == n
    assert dh.nobs == t
    assert dh.nvar == k
    assert_equal(dh.a3d, x.values)
    expected = np.reshape(x.swapaxes(0, 2).values, (n * t, k))
    assert_equal(dh.a2d, expected)
    assert_panel_equal(x, dh.panel)
    expected_frame = x.swapaxes(1, 2).to_frame()
    expected_frame.index.levels[0].name = 'entity'
    expected_frame.index.levels[1].name = 'time'
    assert_frame_equal(dh.dataframe, expected_frame)


def test_pandas_multiindex_dataframe():
    n, t, k = 11, 7, 3
    x = np.random.random((n, t, k))
    major = pd.date_range('12-31-1999', periods=7)
    minor = ['var.{0}'.format(i) for i in range(1, k + 1)]
    items = ['item.{0}'.format(i) for i in range(1, n + 1)]
    x = pd.Panel(x, items=items, major_axis=major, minor_axis=minor)
    x = x.to_frame()
    PanelData(x)


def test_pandas_dataframe():
    t, n = 11, 7
    x = np.random.random((t, n))
    index = pd.date_range('12-31-1999', periods=t)
    cols = ['entity.{0}'.format(i) for i in range(1, n + 1)]
    x = pd.DataFrame(x, columns=cols, index=index)
    PanelData(x)


def test_existing_panel_data():
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    major = pd.date_range('12-31-1999', periods=7)
    items = ['var.{0}'.format(i) for i in range(1, k + 1)]
    minor = ['entities.{0}'.format(i) for i in range(1, n + 1)]
    x = pd.Panel(x, items=items, major_axis=major, minor_axis=minor)
    dh = PanelData(x)
    dh2 = PanelData(dh)
    assert_frame_equal(dh.dataframe, dh2.dataframe)


def test_xarray_2d():
    n, t, k = 11, 7, 1
    x = np.random.random((t, n))
    x = xr.DataArray(x, dims=('time', 'entity'),
                     coords={'entity': list('firm.' + str(i) for i in range(n))})
    dh = PanelData(x)
    assert_equal(dh.a2d, np.reshape(x.values.T, (n * t, 1)))


def test_xarray_3d():
    n, t, k = 11, 7, 13
    x = np.random.random((k, t, n))
    x = xr.DataArray(x, dims=('var', 'time', 'entity'),
                     coords={'entity': list('firm.' + str(i) for i in range(n)),
                             'var': list('x.' + str(i) for i in range(k))})
    dh = PanelData(x)
    assert_equal(np.reshape(x.values.T, (n * t, k)), dh.a2d)


def test_dimensions(panel):
    dh = PanelData(panel)
    assert dh.nentity == panel.shape[2]
    assert dh.nvar == panel.shape[0]
    assert dh.nobs == panel.shape[1]


def test_drop(panel):
    dh = PanelData(panel)
    orig = dh.dataframe.copy()
    sel = np.zeros(orig.shape[0], dtype=np.bool)
    sel[::3] = True
    dh.drop(sel)
    assert dh.dataframe.shape[0] == len(sel) - sel.sum()


def test_labels(panel):
    dh = PanelData(panel)
    assert dh.vars == list(panel.items)
    assert dh.time == list(panel.major_axis)
    assert dh.entities == list(panel.minor_axis)


def test_missing(panel):
    panel.iloc[0, :, ::3] = np.nan
    dh = PanelData(panel)
    assert_equal(dh.isnull, np.any(np.isnan(dh.a2d), 1))


def test_incorrect_dataframe():
    grouped = np.array(list([i] * 10 for i in range(10))).ravel()
    df = pd.DataFrame({'a': np.arange(100),
                       'b': grouped,
                       'c': np.random.permutation(grouped),
                       'data': np.random.randn(100)})
    df = df.set_index(['a', 'b', 'c'])
    with pytest.raises(ValueError):
        PanelData(df)


def test_incorrect_types():
    with pytest.raises(ValueError):
        PanelData(xr.DataArray(np.random.randn(10)))
    with pytest.raises(TypeError):
        PanelData(list(np.random.randn(10)))


def test_ids(panel):
    data = PanelData(panel)
    eids = data.entity_ids
    assert eids.shape == (77, 1)
    assert len(np.unique(eids)) == 11
    for i in range(0, len(eids), 7):
        assert np.ptp(eids[i:i + 7]) == 0
        assert np.all((eids[i + 8:] - eids[i]) != 0)

    tids = data.time_ids
    assert tids.shape == (77, 1)
    assert len(np.unique(tids)) == 7
    for i in range(0, 11):
        assert np.ptp(tids[i::7]) == 0


def test_str_repr_smoke(panel):
    data = PanelData(panel)
    assert 'PanelData' in str(data)
    assert str(hex(id(data))) in data.__repr__()


def test_demean(panel):
    data = PanelData(panel)
    fe = data.demean('entity')
    expected = panel.values.copy()
    for i in range(3):
        expected[i] -= expected[i].mean(0)
    assert_allclose(fe.a3d, expected)

    te = data.demean('time')
    expected = panel.values.copy()
    for i in range(3):
        expected[i] -= expected[i].mean(1)[:, None]
    assert_allclose(te.a3d, expected)


def test_demean_against_groupby(data):
    dh = PanelData(data.x)
    df = dh.dataframe

    def demean(x):
        return x - x.mean()

    entity_demean = df.groupby(level=0).transform(demean)
    res = dh.demean('entity')
    assert_allclose(entity_demean.values, res.dataframe.values)

    time_demean = df.groupby(level=1).transform(demean)
    res = dh.demean('time')
    assert_allclose(time_demean.values, res.dataframe.values)


def test_demean_against_dummy_regression(data):
    dh = PanelData(data.x)
    dh.drop(dh.isnull)

    df = dh.dataframe
    no_index = df.reset_index()

    cat = pd.Categorical(no_index[df.index.levels[0].name])
    d = pd.get_dummies(cat, drop_first=False).astype(np.float64)
    dummy_demeaned = df.values - d @ pinv(d) @ df.values
    entity_demean = dh.demean('entity')
    assert_allclose(1 + np.abs(entity_demean.dataframe.values),
                    1 + np.abs(dummy_demeaned))

    cat = pd.Categorical(no_index[df.index.levels[1].name])
    d = pd.get_dummies(cat, drop_first=False).astype(np.float64)
    dummy_demeaned = df.values - d @ pinv(d) @ df.values
    time_demean = dh.demean('time')
    assert_allclose(1 + np.abs(time_demean.dataframe.values),
                    1 + np.abs(dummy_demeaned))

    cat = pd.Categorical(no_index[df.index.levels[0].name])
    d1 = pd.get_dummies(cat, drop_first=False).astype(np.float64)
    cat = pd.Categorical(no_index[df.index.levels[1].name])
    d2 = pd.get_dummies(cat, drop_first=True).astype(np.float64)
    d = np.c_[d1.values, d2.values]
    dummy_demeaned = df.values - d @ pinv(d) @ df.values
    both_demean = dh.demean('both')
    assert_allclose(1 + np.abs(both_demean.dataframe.values),
                    1 + np.abs(dummy_demeaned))


def test_demean_missing(panel):
    panel.values.flat[::13] = np.nan
    data = PanelData(panel)
    fe = data.demean('entity')
    expected = panel.values.copy()
    for i in range(3):
        expected[i] -= np.nanmean(expected[i], 0)
    assert_allclose(fe.a3d, expected)

    te = data.demean('time')
    expected = panel.values.copy()
    for i in range(3):
        expected[i] -= np.nanmean(expected[i], 1)[:, None]
    assert_allclose(te.a3d, expected)


def test_demean_many_missing(panel):
    panel.iloc[0, ::3] = np.nan
    panel.iloc[0, :, ::3] = np.nan
    panel.iloc[1, ::5] = np.nan
    panel.iloc[1, :, ::5] = np.nan
    panel.iloc[2, ::2] = np.nan
    panel.iloc[2, :, ::2] = np.nan
    data = PanelData(panel)
    fe = data.demean('entity')
    orig_nan = np.isnan(panel.values.ravel())
    fe_nan = np.isnan(fe.a3d.ravel())
    assert np.all(fe_nan[orig_nan])
    expected = panel.values.copy()
    for i in range(3):
        expected[i] -= np.nanmean(expected[i], 0)
    assert_allclose(fe.a3d, expected)

    te = data.demean('time')
    expected = panel.values.copy()
    for i in range(3):
        expected[i] -= np.nanmean(expected[i], 1)[:, None]
    assert_allclose(te.a3d, expected)


def test_demean_many_missing_dropped(panel):
    panel.iloc[0, ::3, ::3] = np.nan
    data = PanelData(panel)
    data.drop(data.isnull)
    fe = data.demean('entity')

    expected = data.a2d.copy()
    eid = data.entity_ids.ravel()
    for i in np.unique(eid):
        expected[eid == i] -= np.nanmean(expected[eid == i], 0)

    assert_allclose(fe.a2d, expected)


def test_demean_both_large_t():
    data = PanelData(pd.Panel(np.random.standard_normal((1, 100, 10))))
    demeaned = data.demean('both')

    df = data.dataframe
    no_index = df.reset_index()
    cat = pd.Categorical(no_index[df.index.levels[0].name])
    d1 = pd.get_dummies(cat, drop_first=False).astype(np.float64)
    cat = pd.Categorical(no_index[df.index.levels[1].name])
    d2 = pd.get_dummies(cat, drop_first=True).astype(np.float64)
    d = np.c_[d1.values, d2.values]
    dummy_demeaned = df.values - d @ pinv(d) @ df.values
    assert_allclose(1 + np.abs(demeaned.dataframe.values),
                    1 + np.abs(dummy_demeaned))


def test_demean_invalid(panel):
    data = PanelData(panel)
    with pytest.raises(ValueError):
        data.demean('unknown')
