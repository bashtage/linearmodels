import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import stats

from linearmodels.utility import InapplicableTestStatistic, InvalidTestStatistic, \
    WaldTestStatistic, cached_property, has_constant, inv_sqrth, ensure_unique_column


def test_hasconstant():
    x = np.random.randn(100, 3)
    hc, loc = has_constant(x)
    assert hc is False
    assert loc is None
    x[:, 0] = 1
    hc, loc = has_constant(x)
    assert hc is True
    assert loc == 0
    x[:, 0] = 2
    hc, loc = has_constant(x)
    assert hc is True
    assert loc == 0
    x[::2, 0] = 0
    x[:, 1] = 1
    x[1::2, 1] = 0
    hc, loc = has_constant(x)
    assert hc is True


def test_wald_statistic():
    ts = WaldTestStatistic(1.0, "_NULL_", 1, name="_NAME_")
    assert str(hex(id(ts))) in ts.__repr__()
    assert '_NULL_' in str(ts)
    assert ts.stat == 1.0
    assert ts.df == 1
    assert ts.df_denom is None
    assert ts.dist_name == 'chi2(1)'
    assert isinstance(ts.critical_values, dict)
    assert_allclose(1 - stats.chi2.cdf(1.0, 1), ts.pval)

    ts = WaldTestStatistic(1.0, "_NULL_", 1, 1000, name="_NAME_")
    assert ts.df == 1
    assert ts.df_denom == 1000
    assert ts.dist_name == 'F(1,1000)'
    assert_allclose(1 - stats.f.cdf(1.0, 1, 1000), ts.pval)


def test_invalid_test_statistic():
    ts = InvalidTestStatistic('_REASON_', name='_NAME_')
    assert str(hex(id(ts))) in ts.__repr__()
    assert '_REASON_' in str(ts)
    assert np.isnan(ts.pval)
    assert ts.critical_values is None


def test_inapplicable_test_statistic():
    ts = InapplicableTestStatistic(reason='_REASON_', name='_NAME_')
    assert str(hex(id(ts))) in ts.__repr__()
    assert '_REASON_' in str(ts)
    assert np.isnan(ts.pval)
    assert ts.critical_values is None

    ts = InapplicableTestStatistic()
    assert 'not applicable' in str(ts)


def test_inv_sqrth():
    x = np.random.randn(1000, 10)
    xpx = x.T @ x
    invsq = inv_sqrth(xpx)
    prod = invsq @ xpx @ invsq - np.eye(10)
    assert_allclose(1 + prod, np.ones((10, 10)))


def test_cached_property():
    class A(object):
        def __init__(self):
            self.a_count = 0

        @cached_property
        def a(self):
            print('a called')
            self.a_count += 1
            return 'a'

    o = A()
    o.__getattribute__('a')
    assert o.a == 'a'
    assert o.a_count == 1
    assert o.a == 'a'
    assert o.a_count == 1
    delattr(o, 'a')
    assert o.a == 'a'
    assert o.a_count == 2

    # To improve coverage
    cp = cached_property(lambda x: x)
    cp.__get__(cp, None)


def test_ensure_unique_column():
    df = pd.DataFrame({'a': [0, 1, 0], 'b': [1.0, 0.0, 1.0]})
    out = ensure_unique_column('a', df)
    assert out == '_a_'
    out = ensure_unique_column('c', df)
    assert out == 'c'
    out = ensure_unique_column('a', df, '=')
    assert out == '=a='
    df['_a_'] = -1
    out = ensure_unique_column('a', df)
    assert out == '__a__'
