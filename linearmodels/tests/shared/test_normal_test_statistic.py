"""
Tests of NormalTestStatistic and of the other holders of test statistics

The p-values and the critical values are checked against each other and against
the normal distribution and the chi-squared distribution with one degree of
freedom, and the objects are checked to survive being pickled and copied, which
is needed to send results to other processes.
"""

import copy
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
    NormalTestStatistic,
    WaldTestStatistic,
)
from linearmodels.tests._utility import assert_same_text

SIZES = {"10%": 0.10, "5%": 0.05, "1%": 0.01}
PROTOCOLS = range(pickle.HIGHEST_PROTOCOL + 1)


@pytest.mark.parametrize("two_sided", [True, False])
def test_critical_values_are_quantiles_of_the_normal(two_sided):
    crit = NormalTestStatistic(0.0, "_NULL_", two_sided=two_sided).critical_values
    assert list(crit) == list(SIZES)
    # The test rejects when the statistic exceeds the critical value in
    # absolute value if two-sided and when it is large if one-sided
    tail = np.array(list(SIZES.values())) / (2 if two_sided else 1)
    assert_allclose(list(crit.values()), stats.norm.isf(tail), rtol=1e-12)
    assert_allclose(list(crit.values()), stats.norm.ppf(1 - tail), rtol=1e-12)
    # Smaller sizes have larger critical values
    assert list(crit.values()) == sorted(crit.values())


@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("label", list(SIZES))
def test_p_value_at_the_critical_value_is_the_size(label, two_sided):
    crit = NormalTestStatistic(0.0, "_NULL_", two_sided=two_sided).critical_values
    at_critical_value = NormalTestStatistic(crit[label], "_NULL_", two_sided=two_sided)
    assert_allclose(at_critical_value.pval, SIZES[label], rtol=1e-10)
    # Statistics beyond the critical value reject
    for sign in (1, -1) if two_sided else (1,):
        beyond = NormalTestStatistic(
            sign * (crit[label] + 0.01), "_NULL_", two_sided=two_sided
        )
        assert beyond.pval < SIZES[label]


def test_one_sided_critical_values_are_smaller_than_two_sided():
    one = NormalTestStatistic(0.0, "_NULL_", two_sided=False).critical_values
    two = NormalTestStatistic(0.0, "_NULL_", two_sided=True).critical_values
    for label in SIZES:
        assert one[label] < two[label]
    assert_allclose(one["10%"], stats.norm.isf(0.1))
    assert_allclose(two["5%"], stats.norm.isf(0.025))


@pytest.mark.parametrize("stat", [-4.0, -1.3, 0.0, 0.2, 1.3, 4.0])
def test_two_sided_p_value_is_a_chi2_p_value_and_symmetric(stat):
    ts = NormalTestStatistic(stat, "_NULL_")
    wald = WaldTestStatistic(stat**2, "_NULL_", 1)
    assert_allclose(ts.pval, wald.pval, rtol=1e-10)
    assert_allclose(NormalTestStatistic(-stat, "_NULL_").pval, ts.pval, rtol=1e-12)
    assert 0 < ts.pval <= 1


@pytest.mark.parametrize("stat", [-4.0, -1.3, 0.0, 0.2, 1.3, 4.0])
def test_one_sided_p_value_is_for_a_large_statistic(stat):
    ts = NormalTestStatistic(stat, "_NULL_", two_sided=False)
    assert_allclose(ts.pval, stats.norm.sf(stat), rtol=1e-12)
    flipped = NormalTestStatistic(-stat, "_NULL_", two_sided=False)
    assert_allclose(ts.pval + flipped.pval, 1.0, rtol=1e-12)
    larger = NormalTestStatistic(stat + 0.5, "_NULL_", two_sided=False)
    assert larger.pval < ts.pval


def test_nonfinite_statistics():
    ts = NormalTestStatistic(np.nan, "_NULL_")
    assert np.isnan(ts.pval)
    assert "Statistic: nan" in str(ts)
    assert "P-value: nan" in str(ts)
    # The critical values do not depend on the statistic
    assert_allclose(
        list(ts.critical_values.values()),
        list(NormalTestStatistic(1.0, "_NULL_").critical_values.values()),
    )
    for stat, two_sided, pval in (
        (np.inf, True, 0.0),
        (-np.inf, True, 0.0),
        (np.inf, False, 0.0),
        (-np.inf, False, 1.0),
    ):
        ts = NormalTestStatistic(stat, "_NULL_", two_sided=two_sided)
        assert_allclose(ts.pval, pval, atol=1e-12)


@pytest.mark.parametrize(
    "stat",
    [np.float64(2.0), np.array(2.0), 2, 2.0],
    ids=["numpy-scalar", "0-d-array", "int", "float"],
)
def test_numpy_and_integer_statistics(stat):
    ts = NormalTestStatistic(stat, "_NULL_", name="_NAME_")
    assert_allclose(ts.pval, 2 * stats.norm.sf(2.0), rtol=1e-12)
    assert "Statistic: 2.0000" in str(ts)


def test_str_and_repr_of_one_sided_statistic():
    ts = NormalTestStatistic(1.5, "_NULL_", name="_NAME_", two_sided=False)
    expected = (
        "_NAME_\nH0: _NULL_\nStatistic: 1.5000\nP-value: 0.0668\nDistributed: N(0,1)"
    )
    assert str(ts) == expected
    assert repr(ts) == expected + "\nNormalTestStatistic, id: " + hex(id(ts))
    unnamed = NormalTestStatistic(-1.5, "_NULL_", two_sided=False)
    assert str(unnamed) == (
        "H0: _NULL_\nStatistic: -1.5000\nP-value: 0.9332\nDistributed: N(0,1)"
    )


def test_str_and_repr_of_two_sided_statistic_with_a_small_p_value():
    ts = NormalTestStatistic(-9.0, "_NULL_", name="_NAME_")
    assert "P-value: 0.0000\n" in str(ts)
    assert str(ts).startswith("_NAME_\nH0: _NULL_\nStatistic: -9.0000\n")
    assert repr(ts).startswith(str(ts))
    assert repr(ts).endswith(f"NormalTestStatistic, id: {hex(id(ts))}")


CASES = {
    "normal": NormalTestStatistic(1.5, "_NULL_", name="_NAME_"),
    "normal-one-sided": NormalTestStatistic(
        -1.5, "_NULL_", name="_NAME_", two_sided=False
    ),
    "normal-nan": NormalTestStatistic(np.nan, "_NULL_"),
    "wald-chi2": WaldTestStatistic(3.0, "_NULL_", 2, name="_NAME_"),
    "wald-f": WaldTestStatistic(3.0, "_NULL_", 2, 40),
    "invalid": InvalidTestStatistic("_REASON_", name="_NAME_"),
    "inapplicable": InapplicableTestStatistic(reason="_REASON_", name="_NAME_"),
    "inapplicable-default": InapplicableTestStatistic(),
}


def assert_same_statistic(clone, original):
    assert type(clone) is type(original)
    assert clone.dist_name == original.dist_name
    if hasattr(original, "df"):
        assert clone.df == original.df
        assert clone.df_denom == original.df_denom
    assert_same_text(str(clone), str(original))
    assert_allclose(clone.stat, original.stat)
    assert_allclose(clone.pval, original.pval)
    assert clone.null == original.null
    if original.critical_values is None:
        assert clone.critical_values is None
    else:
        assert clone.critical_values.keys() == original.critical_values.keys()
        assert_allclose(
            list(clone.critical_values.values()),
            list(original.critical_values.values()),
        )
    # The distribution that is used to compute the p-value is usable
    assert_allclose(
        clone.dist.sf(1.25), original.dist.sf(1.25), rtol=1e-12, equal_nan=True
    )


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("name", list(CASES))
def test_pickle_round_trip(name, protocol):
    original = CASES[name]
    clone = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert clone is not original
    assert_same_statistic(clone, original)
    # The hex id in the repr is the id of the object
    assert repr(clone).endswith(hex(id(clone)))


@pytest.mark.parametrize("name", list(CASES))
def test_copy_and_deepcopy(name):
    original = CASES[name]
    assert_same_statistic(copy.copy(original), original)
    assert_same_statistic(copy.deepcopy(original), original)


def test_pickle_preserves_the_side_of_the_test():
    two_sided = pickle.loads(pickle.dumps(NormalTestStatistic(1.5, "_NULL_")))
    one_sided = pickle.loads(
        pickle.dumps(NormalTestStatistic(1.5, "_NULL_", two_sided=False))
    )
    assert_allclose(two_sided.pval, 2 * stats.norm.sf(1.5))
    assert_allclose(one_sided.pval, stats.norm.sf(1.5))
    assert one_sided.critical_values["5%"] < two_sided.critical_values["5%"]
