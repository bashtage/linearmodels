from linearmodels.compat.numpy import isin
from linearmodels.compat.pandas import concat

import numpy as np
from numpy.testing import assert_array_equal
from pandas import DataFrame, Series, date_range
import pytest

from linearmodels.utility import AttrDict


@pytest.fixture('module')
def data():
    idx = date_range('2000-01-01', periods=100)
    df1 = DataFrame(np.arange(100)[:, None], columns=['A'], index=idx)
    x = np.reshape(np.arange(200), (100, 2))
    df2 = DataFrame(x, columns=['B', 'C'], index=idx[::-1])
    s = Series(300 + np.arange(100), index=idx, name='D')
    return AttrDict(df1=df1, df2=df2, s=s)


def test_concat_sort(data):
    a = concat([data.df1, data.df2], 1)
    b = concat([data.df1, data.df2, data.s], 1)
    c = concat([data.df1, data.df2, data.s], 1, sort=True)
    d = concat([data.df2, data.df1, data.s], 1, sort=False)
    assert list(a.columns) == ['A', 'B', 'C']
    assert list(b.columns) == ['A', 'B', 'C', 'D']
    assert list(c.columns) == ['A', 'B', 'C', 'D']
    assert list(d.columns) == ['B', 'C', 'A', 'D']


def test_isin():
    a = np.arange(5)
    b = np.arange(3)
    expected = np.array([1, 1, 1, 0, 0], dtype=np.bool)
    assert_array_equal(isin(a, b), np.array([1, 1, 1, 0, 0], dtype=np.bool))

    a = np.column_stack([a] * 3)
    expected = np.column_stack([expected] * 3)
    assert_array_equal(isin(a, b), expected)
