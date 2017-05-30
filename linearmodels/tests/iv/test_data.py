from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_equal

from linearmodels.compat.pandas import assert_frame_equal, assert_series_equal
from linearmodels.iv.data import IVData


class TestDataHandler(object):
    def test_numpy_2d(self):
        x = np.empty((10, 2))
        xdh = IVData(x)
        assert xdh.ndim == x.ndim
        assert xdh.cols == ['x.0', 'x.1']
        assert xdh.rows == list(np.arange(10))
        assert_equal(xdh.ndarray, x)
        df = pd.DataFrame(x, columns=xdh.cols, index=xdh.rows)
        assert_frame_equal(xdh.pandas, df)
        assert xdh.shape == (10, 2)
        assert xdh.labels == {0: xdh.rows, 1: xdh.cols}

    def test_numpy_1d(self):
        x = np.empty(10)
        xdh = IVData(x)
        assert xdh.ndim == 2
        assert xdh.cols == ['x']
        assert xdh.rows == list(np.arange(10))
        assert_equal(xdh.ndarray, x[:, None])
        df = pd.DataFrame(x[:, None], columns=xdh.cols, index=xdh.rows)
        assert_frame_equal(xdh.pandas, df)
        assert xdh.shape == (10, 1)

    def test_pandas_df_numeric(self):
        x = np.empty((10, 2))
        index = pd.date_range('2017-01-01', periods=10)
        xdf = pd.DataFrame(x, columns=['a', 'b'], index=index)
        xdh = IVData(xdf)
        assert xdh.ndim == 2
        assert xdh.cols == list(xdf.columns)
        assert xdh.rows == list(xdf.index)
        assert_equal(xdh.ndarray, x)
        df = pd.DataFrame(x, columns=xdh.cols, index=xdh.rows)
        assert_frame_equal(xdh.pandas, df)
        assert xdh.shape == (10, 2)

    def test_pandas_series_numeric(self):
        x = np.empty(10)
        index = pd.date_range('2017-01-01', periods=10)
        xs = pd.Series(x, name='charlie', index=index)
        xdh = IVData(xs)
        assert xdh.ndim == 2
        assert xdh.cols == [xs.name]
        assert xdh.rows == list(xs.index)
        assert_equal(xdh.ndarray, x[:, None])
        df = pd.DataFrame(x[:, None], columns=xdh.cols, index=xdh.rows)
        assert_frame_equal(xdh.pandas, df)
        assert xdh.shape == (10, 1)

    def test_xarray_1d(self):
        x_np = np.random.randn(10)
        x = xr.DataArray(x_np)
        dh = IVData(x, 'some_variable')
        assert_equal(dh.ndarray, x_np[:, None])
        assert dh.rows == list(np.arange(10))
        assert dh.cols == ['some_variable.0']
        expected = pd.DataFrame(x_np, columns=dh.cols, index=dh.rows)
        assert_frame_equal(expected, dh.pandas)

        index = pd.date_range('2017-01-01', periods=10)
        x = xr.DataArray(x_np,
                         [('time', index)])
        dh = IVData(x, 'some_variable')
        assert_equal(dh.ndarray, x_np[:, None])
        assert_series_equal(pd.Series(dh.rows), pd.Series(list(index)))
        assert dh.cols == ['some_variable.0']
        expected = pd.DataFrame(x_np[:, None], columns=dh.cols, index=dh.rows)
        assert_frame_equal(expected, dh.pandas)

    def test_xarray_2d(self):
        x_np = np.random.randn(10, 2)
        x = xr.DataArray(x_np)
        dh = IVData(x)
        assert_equal(dh.ndarray, x_np)
        assert dh.rows == list(np.arange(10))
        assert dh.cols == ['x.0', 'x.1']
        expected = pd.DataFrame(x_np, columns=dh.cols, index=dh.rows)
        assert_frame_equal(expected, dh.pandas)

        index = pd.date_range('2017-01-01', periods=10)
        x = xr.DataArray(x_np,
                         [('time', index),
                          ('variables', ['apple', 'banana'])])
        dh = IVData(x)
        assert_equal(dh.ndarray, x_np)
        assert_series_equal(pd.Series(dh.rows), pd.Series(list(index)))
        assert dh.cols == ['apple', 'banana']
        expected = pd.DataFrame(x_np, columns=dh.cols, index=dh.rows)
        assert_frame_equal(expected, dh.pandas)

    def test_invalid_types(self):
        with pytest.raises(ValueError):
            IVData(np.empty((1, 1, 1)))
        with pytest.raises(ValueError):
            IVData(np.empty((10, 2, 2)))
        with pytest.raises(TypeError):
            class a(object):
                @property
                def ndim(self):
                    return 2

            IVData(a())

    def test_string_cat_equiv(self):
        s1 = pd.Series(['a', 'b', 'a', 'b', 'c', 'd', 'a', 'b'])
        s2 = pd.Series(np.arange(8.0))
        s3 = pd.Series(['apple', 'banana', 'apple', 'banana',
                        'cherry', 'date', 'apple', 'banana'])
        df = pd.DataFrame({'string': s1, 'number': s2, 'other_string': s3})
        dh = IVData(df)
        df_cat = df.copy()
        df_cat['string'] = df_cat['string'].astype('category')
        dh_cat = IVData(df_cat)
        assert_frame_equal(dh.pandas, dh_cat.pandas)

    def test_existing_datahandler(self):
        x = np.empty((10, 2))
        index = pd.date_range('2017-01-01', periods=10)
        xdf = pd.DataFrame(x, columns=['a', 'b'], index=index)
        xdh = IVData(xdf)
        xdh2 = IVData(xdh)
        assert xdh is not xdh2
        assert xdh.cols == xdh2.cols
        assert xdh.rows == xdh2.rows
        assert_equal(xdh.ndarray, xdh2.ndarray)
        assert xdh.ndim == xdh2.ndim
        assert_frame_equal(xdh.pandas, xdh2.pandas)

    def test_categorical(self):
        index = pd.date_range('2017-01-01', periods=10)
        cat = pd.Categorical(['a', 'b', 'a', 'b', 'a', 'a', 'b', 'c', 'c', 'a'])
        num = np.empty(10)
        df = pd.DataFrame(OrderedDict(cat=cat, num=num), index=index)
        dh = IVData(df)
        assert dh.ndim == 2
        assert dh.shape == (10, 3)
        assert sorted(dh.cols) == sorted(['cat.b', 'cat.c', 'num'])
        assert dh.rows == list(index)
        assert_equal(dh.pandas['num'].values, num)
        assert_equal(dh.pandas['cat.b'].values, (cat == 'b').astype(np.float))
        assert_equal(dh.pandas['cat.c'].values, (cat == 'c').astype(np.float))

    def test_categorical_series(self):
        index = pd.date_range('2017-01-01', periods=10)
        cat = pd.Categorical(['a', 'b', 'a', 'b', 'a', 'a', 'b', 'c', 'c', 'a'])
        s = pd.Series(cat, name='cat', index=index)
        dh = IVData(s)
        assert dh.ndim == 2
        assert dh.shape == (10, 2)
        assert sorted(dh.cols) == sorted(['cat.b', 'cat.c'])
        assert dh.rows == list(index)
        assert_equal(dh.pandas['cat.b'].values, (cat == 'b').astype(np.float))
        assert_equal(dh.pandas['cat.c'].values, (cat == 'c').astype(np.float))

    def test_categorical_no_conversion(self):
        index = pd.date_range('2017-01-01', periods=10)
        cat = pd.Categorical(['a', 'b', 'a', 'b', 'a', 'a', 'b', 'c', 'c', 'a'])
        s = pd.Series({'cat': cat}, index=index, name='cat')
        dh = IVData(s, convert_dummies=False)
        assert dh.ndim == 2
        assert dh.shape == (10, 1)
        assert dh.cols == ['cat']
        assert dh.rows == list(index)
        df = pd.DataFrame(s)
        assert_frame_equal(dh.pandas, df)

    def test_categorical_keep_first(self):
        index = pd.date_range('2017-01-01', periods=10)
        cat = pd.Categorical(['a', 'b', 'a', 'b', 'a', 'a', 'b', 'c', 'c', 'a'])
        num = np.empty(10)
        df = pd.DataFrame(OrderedDict(cat=cat, num=num), index=index)
        dh = IVData(df, drop_first=False)
        assert dh.ndim == 2
        assert dh.shape == (10, 4)
        assert sorted(dh.cols) == sorted(['cat.a', 'cat.b', 'cat.c', 'num'])
        assert dh.rows == list(index)
        assert_equal(dh.pandas['num'].values, num)
        assert_equal(dh.pandas['cat.a'].values, (cat == 'a').astype(np.float))
        assert_equal(dh.pandas['cat.b'].values, (cat == 'b').astype(np.float))
        assert_equal(dh.pandas['cat.c'].values, (cat == 'c').astype(np.float))

    def test_nobs_missing_error(self):
        with pytest.raises(ValueError):
            IVData(None)

    def test_incorrect_nobs(self):
        x = np.empty((10, 1))
        with pytest.raises(ValueError):
            IVData(x, nobs=100)

    def test_mixed_data(self):
        s = pd.Series([1, 2, 'a', -3.0])
        with pytest.raises(ValueError):
            IVData(s)
