import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import linearmodels
from linearmodels.utility import (InapplicableTestStatistic,
                                  InvalidTestStatistic, WaldTestStatistic,
                                  cached_property, ensure_unique_column,
                                  has_constant, inv_sqrth, AttrDict,
                                  missing_warning)


def test_missing_warning():
    missing = np.zeros(500, dtype=np.bool)
    with warnings.catch_warnings(record=True) as w:
        missing_warning(missing)
        assert len(w) == 0

    missing[0] = True
    with warnings.catch_warnings(record=True) as w:
        missing_warning(missing)
        assert len(w) == 1

    original = linearmodels.WARN_ON_MISSING
    linearmodels.WARN_ON_MISSING = False
    with warnings.catch_warnings(record=True) as w:
        missing_warning(missing)
        assert len(w) == 0
    linearmodels.WARN_ON_MISSING = original


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


def test_attr_dict():
    ad = AttrDict()
    ad['one'] = 'one'
    ad[1] = 1
    ad[('a', 2)] = ('a', 2)
    assert list(ad.keys()) == ['one', 1, ('a', 2)]
    assert len(ad) == 3

    ad2 = ad.copy()
    assert list(ad2.keys()) == list(ad.keys())
    assert ad.get('one', None) == 'one'
    assert ad.get('two', False) is False

    k, v = ad.popitem()
    assert k == 'one'
    assert v == 'one'

    items = ad.items()
    assert (1, 1) in items
    assert (('a', 2), ('a', 2)) in items
    assert len(items) == 2

    values = ad.values()
    assert 1 in values
    assert ('a', 2) in values
    assert len(values) == 2

    ad2 = AttrDict()
    ad2[1] = 3
    ad2['one'] = 'one'
    ad2['a'] = 'a'
    ad.update(ad2)
    assert ad[1] == 3
    assert 'a' in ad

    ad.__str__()
    with pytest.raises(AttributeError):
        ad.__ordered_dict__ = None
    with pytest.raises(AttributeError):
        ad.some_other_key
    with pytest.raises(KeyError):
        ad['__ordered_dict__'] = None

    del ad[1]
    assert 1 not in ad.keys()
    ad.new_value = 'new_value'
    assert 'new_value' in ad.keys()
    assert ad.new_value == ad['new_value']

    for key in ad.keys():
        if isinstance(key, str):
            assert key in dir(ad)

    new_value = ad.pop('new_value')
    assert new_value == 'new_value'

    del ad.one
    assert 'one' not in ad.keys()

    ad.clear()
    assert list(ad.keys()) == []
