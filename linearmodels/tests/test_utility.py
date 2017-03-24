import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from linearmodels.utility import WaldTestStatistic, InvalidTestStatistic, \
    has_constant, inv_sqrth


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
    print(loc)


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


def test_inv_sqrth():
    x = np.random.randn(1000, 10)
    xpx = x.T @ x
    invsq = inv_sqrth(xpx)
    prod = invsq @ xpx @ invsq - np.eye(10)
    assert_allclose(1 + prod, np.ones((10, 10)))
