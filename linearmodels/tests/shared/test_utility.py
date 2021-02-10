import pickle
import random
import string
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats

import linearmodels
from linearmodels.shared.exceptions import missing_warning
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
    WaldTestStatistic,
)
from linearmodels.shared.io import add_star, format_wide
from linearmodels.shared.linalg import has_constant, inv_sqrth
from linearmodels.shared.utility import AttrDict, ensure_unique_column, panel_to_frame

MISSING_PANEL = "Panel" not in dir(pd)


def test_missing_warning():
    missing = np.zeros(500, dtype=bool)
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
    assert bool(hc) is False
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
    assert "_NULL_" in str(ts)
    assert ts.stat == 1.0
    assert ts.df == 1
    assert ts.df_denom is None
    assert ts.dist_name == "chi2(1)"
    assert isinstance(ts.critical_values, dict)
    assert_allclose(1 - stats.chi2.cdf(1.0, 1), ts.pval)

    ts = WaldTestStatistic(1.0, "_NULL_", 1, 1000, name="_NAME_")
    assert ts.df == 1
    assert ts.df_denom == 1000
    assert ts.dist_name == "F(1,1000)"
    assert_allclose(1 - stats.f.cdf(1.0, 1, 1000), ts.pval)


def test_invalid_test_statistic():
    ts = InvalidTestStatistic("_REASON_", name="_NAME_")
    assert str(hex(id(ts))) in ts.__repr__()
    assert "_REASON_" in str(ts)
    assert np.isnan(ts.pval)
    assert ts.critical_values is None


def test_inapplicable_test_statistic():
    ts = InapplicableTestStatistic(reason="_REASON_", name="_NAME_")
    assert str(hex(id(ts))) in ts.__repr__()
    assert "_REASON_" in str(ts)
    assert np.isnan(ts.pval)
    assert ts.critical_values is None

    ts = InapplicableTestStatistic()
    assert "not applicable" in str(ts)


def test_inv_sqrth():
    x = np.random.randn(1000, 10)
    xpx = x.T @ x
    invsq = inv_sqrth(xpx)
    prod = invsq @ xpx @ invsq - np.eye(10)
    assert_allclose(1 + prod, np.ones((10, 10)))


def test_ensure_unique_column():
    df = pd.DataFrame({"a": [0, 1, 0], "b": [1.0, 0.0, 1.0]})
    out = ensure_unique_column("a", df)
    assert out == "_a_"
    out = ensure_unique_column("c", df)
    assert out == "c"
    out = ensure_unique_column("a", df, "=")
    assert out == "=a="
    df["_a_"] = -1
    out = ensure_unique_column("a", df)
    assert out == "__a__"


def test_attr_dict():
    ad = AttrDict()
    ad["one"] = "one"
    ad[1] = 1
    ad[("a", 2)] = ("a", 2)
    assert list(ad.keys()) == ["one", 1, ("a", 2)]
    assert len(ad) == 3

    plk = pickle.dumps(ad)
    pad = pickle.loads(plk)
    assert list(pad.keys()) == ["one", 1, ("a", 2)]
    assert len(pad) == 3

    ad2 = ad.copy()
    assert list(ad2.keys()) == list(ad.keys())
    assert ad.get("one", None) == "one"
    assert ad.get("two", False) is False

    k, v = ad.popitem()
    assert k == "one"
    assert v == "one"

    items = ad.items()
    assert (1, 1) in items
    assert (("a", 2), ("a", 2)) in items
    assert len(items) == 2

    values = ad.values()
    assert 1 in values
    assert ("a", 2) in values
    assert len(values) == 2

    ad2 = AttrDict()
    ad2[1] = 3
    ad2["one"] = "one"
    ad2["a"] = "a"
    ad.update(ad2)
    assert ad[1] == 3
    assert "a" in ad

    ad.__str__()
    with pytest.raises(AttributeError):
        ad.__private_dict__ = None
    with pytest.raises(AttributeError):
        ad.some_other_key
    with pytest.raises(KeyError):
        ad["__private_dict__"] = None

    del ad[1]
    assert 1 not in ad.keys()
    ad.new_value = "new_value"
    assert "new_value" in ad.keys()
    assert ad.new_value == ad["new_value"]

    for key in ad.keys():
        if isinstance(key, str):
            assert key in dir(ad)

    new_value = ad.pop("new_value")
    assert new_value == "new_value"

    del ad.one
    assert "one" not in ad.keys()

    ad.clear()
    assert list(ad.keys()) == []


def test_format_wide():
    k = 26
    inputs = [chr(65 + i) * (20 + i) for i in range(k)]
    out = format_wide(inputs, 80)
    assert max(map(lambda v: len(v), out)) <= 80


def test_panel_to_midf():
    x = np.random.standard_normal((3, 7, 100))
    df = panel_to_frame(x, list(range(3)), list(range(7)), list(range(100)))
    mi = pd.MultiIndex.from_product([list(range(7)), list(range(100))])
    expected = pd.DataFrame(index=mi, columns=[0, 1, 2])
    for i in range(3):
        expected[i] = x[i].ravel()
    expected.index.names = ["major", "minor"]
    pd.testing.assert_frame_equal(df, expected)

    expected2 = expected.copy()
    expected2 = expected2.sort_index(level=[1, 0])
    expected2.index = expected2.index.swaplevel(0, 1)
    expected2.index.names = ["major", "minor"]
    df2 = panel_to_frame(x, list(range(3)), list(range(7)), list(range(100)), True)
    pd.testing.assert_frame_equal(df2, expected2)

    entities = list(
        map(
            "".join,
            [
                [random.choice(string.ascii_lowercase) for __ in range(10)]
                for _ in range(100)
            ],
        )
    )
    times = pd.date_range("1999-12-31", freq="A-DEC", periods=7)
    var_names = ["x.{0}".format(i) for i in range(1, 4)]
    df3 = panel_to_frame(x, var_names, times, entities, True)
    mi = pd.MultiIndex.from_product([times, entities])
    expected3 = pd.DataFrame(index=mi, columns=var_names)
    for i in range(1, 4):
        expected3["x.{0}".format(i)] = x[i - 1].ravel()
    expected3.index = expected3.index.swaplevel(0, 1)
    mi = pd.MultiIndex.from_product([entities, times])
    expected3 = expected3.loc[mi]
    expected3.index.names = ["major", "minor"]
    pd.testing.assert_frame_equal(df3, expected3)


@pytest.mark.parametrize("pvalue", [0.2, 0.11, 0.10, 0.050001, 0.05, 0.01, 0.005])
def test_add_star(pvalue):
    if pvalue <= 0.01:
        expected = "***"
    elif pvalue <= 0.05:
        expected = "**"
    elif pvalue <= 0.10:
        expected = "*"
    else:
        expected = ""

    result = add_star("", pvalue, True)
    assert expected == result
