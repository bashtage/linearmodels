import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from pandas.util.testing import assert_frame_equal, assert_panel_equal

from linearmodels.panel.data import PanelDataHandler


@pytest.fixture
def seed():
    np.random.seed(1234)


def test_numpy_3d(seed):
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    dh = PanelDataHandler(x)
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
    expected_frame = expected.swapaxes(1,2).to_frame()
    assert_frame_equal(dh.dataframe, expected_frame)


def test_numpy_1d(seed):
    n, t, k = 11, 1, 1
    x = np.random.random(n)
    with pytest.raises(ValueError):
        PanelDataHandler(x)


def test_numpy_2d(seed):
    n, t, k = 11, 7, 1
    x = np.random.random((t, n))
    dh = PanelDataHandler(x)
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
    expected_frame = expected.swapaxes(1,2).to_frame()
    assert_frame_equal(dh.dataframe, expected_frame)


def test_pandas_panel(seed):
    n, t, k = 11, 7, 3
    x = np.random.random((k, t, n))
    major = pd.date_range('12-31-1999', periods=7)
    items = ['var.{0}'.format(i) for i in range(1, k + 1)]
    minor = ['entities.{0}'.format(i) for i in range(1, n + 1)]
    x = pd.Panel(x, items=items, major_axis=major, minor_axis=minor)
    dh = PanelDataHandler(x)
    assert dh.nentity == n
    assert dh.nobs == t
    assert dh.nvar == k
    assert_equal(dh.a3d, x.values)
    expected = np.reshape(x.swapaxes(0,2).values, (n * t, k))
    assert_equal(dh.a2d, expected)
    assert_panel_equal(x, dh.panel)
    expected_frame = x.swapaxes(1,2).to_frame()
    assert_frame_equal(dh.dataframe, expected_frame)


def test_pandas_multiindex_dataframe(seed):
    n, t, k = 11, 7, 3
    x = np.random.random((n, t, k))
    major = pd.date_range('12-31-1999', periods=7)
    minor = ['var.{0}'.format(i) for i in range(1, k + 1)]
    items = ['item.{0}'.format(i) for i in range(1, n + 1)]
    x = pd.Panel(x, items=items, major_axis=major, minor_axis=minor)
    x = x.to_frame()
    PanelDataHandler(x)


def test_pandas_dataframe(seed):
    t, n = 11, 7
    x = np.random.random((t, n))
    index = pd.date_range('12-31-1999', periods=t)
    cols = ['entity.{0}'.format(i) for i in range(1, n + 1)]
    x = pd.DataFrame(x, columns=cols, index=index)
    PanelDataHandler(x)
