from linearmodels.compat.statsmodels import Summary

from linearmodels.shared.base import _SummaryStr


def test_sumary_str():
    ss = _SummaryStr()
    assert isinstance(ss.summary, Summary)
