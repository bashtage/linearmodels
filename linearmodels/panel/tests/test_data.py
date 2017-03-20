import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_equal
from pandas.util.testing import assert_frame_equal, assert_panel_equal

from linearmodels.panel.data import PanelData


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
    grouped = np.array(list([i]*10 for i in range(10))).ravel()
    df = pd.DataFrame({'a':np.arange(100),
                       'b':grouped,
                       'c':np.random.permutation(grouped),
                       'data': np.random.randn(100)})
    df = df.set_index(['a','b','c'])
    with pytest.raises(ValueError):
        PanelData(df)

def test_incorrect_types():
    with pytest.raises(ValueError):
        PanelData(xr.DataArray(np.random.randn(10)))
    with pytest.raises(TypeError):
        PanelData(list(np.random.randn(10)))


