import numpy as np
import pytest
from pandas import DataFrame, Series, date_range

from linearmodels.compat.pandas import concat
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
