"""
Tests of the number formatters of the summaries
"""

import numpy as np
import pytest

from linearmodels.shared.io import _str, pval_format
from linearmodels.system import SUR


@pytest.mark.parametrize("value", [np.inf, np.float64("inf")], ids=["float", "numpy"])
def test_str_positive_infinity(value):
    assert _str(value) == "     inf"


@pytest.mark.parametrize("value", [-np.inf, np.float64("-inf")], ids=["float", "numpy"])
def test_str_negative_infinity(value):
    assert _str(value) == "    -inf"


def test_str_has_the_width_of_the_other_numbers():
    assert len(_str(np.inf)) == len(_str(-np.inf)) == len(_str(1e-5)) == 8
    assert len(_str(np.nan)) == 8


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.nan, "        "),
        (0.0, "0.0000"),
        (1.23456, "1.2346"),
        (-0.5, "-0.5000"),
        (12345.678, "1.235e+04"),
        (123456.789, "1.235e+05"),
        (1e-5, "   1e-05"),
    ],
)
def test_str_finite_values_are_not_changed(value, expected):
    assert _str(value) == expected


def test_pval_format():
    assert pval_format(np.nan) == "        "
    assert pval_format(0.04567) == "0.0457"


def test_summary_with_a_zero_standard_error():
    # An exact fit has a zero standard error and an infinite t-statistic
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 1))
    equations = {
        "a": {"dependent": 2 * x[:, 0], "exog": x},
        "b": {"dependent": x[:, 0] + rng.standard_normal(50), "exog": x},
    }
    res = SUR(equations).fit(method="ols")
    assert np.asarray(res.std_errors)[0] == 0.0
    assert np.isinf(res.tstats.iloc[0])
    summary = str(res.summary)
    assert "inf" in summary
    assert "Equation: a" in summary
